"""
eval_steering_analysis.py: 大規模ステアリング分析実験（改良版v3）

=== 改良点（バグ修正） ===

【問題点 - 旧実装（eval_steering_analysis.py v2）】
1. HiddenStateAnalysisとActivationSteererの2つのhookが混在
   - ActivationSteererはステアリング係数を正しく適用
   - HiddenStateTrackerは活用されていない
   - 結果: steering係数を変更してもスコアが変わらないバグ

2. 投影値の記録が不完全
   - steerer.get_projections() から投影値は取得
   - しかしノルムが全く記録されない（batch_norms は常に空）
   - 生成文章のトークン平均ベクトルが記録されない

3. ステアリング適用の確認不足
   - coef == 0 の場合の処理がない
   - steering係数の影響を検証できない

【解決策 - 新実装（改良版v3）】

1. **ProjectionTrackerベースの統一設計**
   - eval_persona.py の正しいパターンに従う
   - ActivationSteerer のみを使用（単一hook）
   - HiddenStateTracker と ActivationSteerer を組み合わせて使用

2. **ActivationSteerer の正しい使用**
   ```python
   # 重要：coeff を必ず渡す
   steerer = ActivationSteerer(
       model,
       steering_vectors[steering_layer],
       coeff=coef,  # ← ここで係数を正しく渡す
       layer_idx=steering_layer,
       positions="response",
   )

   with steerer:
       outputs = model.generate(...)
   ```

3. **隠れ状態追跡の正確な実装**
   - HiddenStateTracker で全層の隠れ状態を記録
   - 保存対象層（4層に1つ + steering層 + 最終層）のみ保存
   - 全層のノルムと投影値を計算して保存

4. **各層のPersona Vector投影値の計算**
   - トークン平均ベクトルを各層で計算
   - 対応するPersona Vectorに投影
   - jsonl形式で層別に保存

【データ流れ】
- Phase 1: 応答生成（steering適用）+ 隠れ状態記録
  → responses.csv（中間）
- Phase 2: スコア判定（OpenAI API or ローカルLLM）
- Phase 3: 結果保存
  → responses.csv + hidden_states.jsonl

【hidden_states.jsonlの構造】
```json
{
  "sample_idx": 0,
  "hidden_states": {"0": [0.1, 0.2, ...], "4": [...], ...},
  "norms": {"0": 1.234, "1": 1.245, ..., "27": 0.987},
  "projections": {"0": 0.567, "1": 0.234, ..., "27": -0.123}
}
```
"""

import asyncio
import gc
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval.model_utils import load_model
from eval.prompts import Prompts
from src.activation_steer import ActivationSteerer
from src.chat_template_utils import apply_chat_template_safe
from src.config import setup_credentials
from src.hidden_state_tracker import HiddenStateTracker
from src.judge import OpenAiJudge
from src.judge_local import BatchLocalLLMJudge

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

config = setup_credentials()


def get_model_layer_count(model_name: str) -> int:
    """モデルの層数を取得"""
    model_layer_count_map = {
        "Qwen/Qwen2.5-7B-Instruct": 28,
        "meta-llama/Llama-3.1-8B-Instruct": 32,
        "meta-llama/Llama-3.1-70B-Instruct": 80,
        "Mistral-7B-Instruct": 32,
    }
    return model_layer_count_map.get(model_name, 32)


def get_selective_layers_to_save(
    num_layers: int,
    steering_layer: int,
) -> Set[int]:
    """保存すべき層を取得（4層に1つ + ステアリング層 + 最終層）"""
    layers_to_save = set()
    for i in range(1, num_layers + 1):
        if (i - 1) % 4 == 0:
            layers_to_save.add(i - 1)
    layers_to_save.add(steering_layer)
    layers_to_save.add(num_layers - 1)
    return sorted(layers_to_save)


def load_persona_vectors(
    trait: str, model_name: str, vector_type: str = "response_avg_diff"
) -> Tuple[Dict[int, torch.Tensor], int]:
    """Persona Vectorを読み込む"""
    vector_path = f"data/persona_vectors/{model_name}/{trait}_{vector_type}.pt"

    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    vectors = torch.load(vector_path, weights_only=False)

    if isinstance(vectors, torch.Tensor):
        num_layers = vectors.shape[0]
        vectors_dict = {i: vectors[i] for i in range(num_layers)}
        return vectors_dict, num_layers
    elif isinstance(vectors, dict):
        return vectors, len(vectors)
    else:
        raise ValueError(f"Unexpected vector type: {type(vectors)}")


def load_eval_questions(trait: str, version: str = "eval") -> List[Dict]:
    """評価用の質問を読み込む（POS ONLYに変更）"""
    data_path = f"data_generation/trait_data_{version}/{trait}.json"

    with open(data_path, "r") as f:
        trait_data = json.load(f)

    raw_questions = trait_data.get("questions", [])
    instructions = trait_data.get("instruction", [])

    questions_data = []
    for i, question in enumerate(raw_questions):
        for inst_idx, instruction in enumerate(instructions):
            system_prompt = f"You are a {trait} assistant. {instruction['pos']}"

            questions_data.append(
                {
                    "question_id": f"{trait}_{i}_pos_{inst_idx}",
                    "question": question,
                    "system": system_prompt,
                }
            )

    return questions_data


def get_intermediate_response_path(output_path: str) -> str:
    """中間結果ファイルのパスを取得"""
    dir_path = os.path.dirname(output_path)
    file_name = os.path.basename(output_path)
    name, ext = os.path.splitext(file_name)
    return os.path.join(dir_path, f"{name}_responses{ext}")


def generate_responses_with_projections_and_hidden_states(
    model,
    tokenizer,
    questions_data: List[Dict],
    coef: float,
    steering_vectors: Dict[int, torch.Tensor],
    steering_layer: int,
    num_layers: int,
    layers_to_save: Set[int],
    batch_size: int = 100,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    n_per_question: int = 10,
    return_to_original_norm: bool = False,
) -> Tuple[
    List[Dict],
    Dict[int, List[torch.Tensor]],
    Dict[int, List[float]],
    Dict[int, List[float]],
]:
    """投影値とトークン平均ベクトルを記録する統一実装

    このバージョンでは以下を行う：
    1. ActivationSteerer でステアリングを適用して生成
    2. 別途HiddenStateTrackerでステアリングなし/ありの隠れ状態を記録
    3. 各層のノルムと投影値を計算して保存

    問題：steering ありの隠れ状態を記録する場合、
    フックを2つ同時に実行する必要がある
    """
    responses = []
    hidden_states_to_save: Dict[int, List[torch.Tensor]] = {
        layer: [] for layer in layers_to_save
    }
    all_norms: Dict[int, List[float]] = {i: [] for i in range(num_layers)}
    all_projections: Dict[int, List[float]] = {i: [] for i in range(num_layers)}

    all_questions_expanded = []
    for q_data in questions_data:
        for repeat_idx in range(n_per_question):
            expanded_q = q_data.copy()
            expanded_q["repeat_idx"] = repeat_idx
            all_questions_expanded.append(expanded_q)

    print(f"Total conversations to generate: {len(all_questions_expanded)}")

    for batch_start in tqdm(
        range(0, len(all_questions_expanded), batch_size), desc="Generating responses"
    ):
        batch_end = min(batch_start + batch_size, len(all_questions_expanded))
        batch_questions = all_questions_expanded[batch_start:batch_end]

        conversations = []
        for q_data in batch_questions:
            messages = []
            if q_data.get("system"):
                messages.append({"role": "system", "content": q_data["system"]})
            messages.append({"role": "user", "content": q_data["question"]})
            conversations.append(messages)

        texts = []
        for messages in conversations:
            text = apply_chat_template_safe(
                tokenizer, messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        tokenized_batch = tokenizer(texts, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        # HiddenStateTrackerを初期化
        tracker = HiddenStateTracker(model, num_layers, model.config.hidden_size)

        # ActivationSteerer と HiddenStateTracker を両方使用
        steerer = ActivationSteerer(
            model,
            steering_vectors[steering_layer],
            coeff=coef,
            layer_idx=steering_layer,
            positions="response",
            renorm_to_original_norm=return_to_original_norm,
            track_projections=False,
        )

        with torch.no_grad():
            with steerer:
                outputs = model.generate(
                    **tokenized_batch,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 1e-7),
                    top_p=1.0,
                    do_sample=(temperature > 0),
                    use_cache=True,
                )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs
        ]

        # HiddenStateTrackerから隠れ状態を取得
        all_avg_hidden_states = tracker.get_all_average_hidden_states()
        all_batch_norms = tracker.compute_all_norms()
        all_batch_projections = tracker.compute_all_projections_per_layer(
            steering_vectors
        )

        for i, (q_data, response) in enumerate(zip(batch_questions, batch_outputs)):
            response_data = {
                "question_id": q_data["question_id"],
                "question": q_data["question"],
                "system": q_data.get("system", ""),
                "prompt": texts[i],
                "answer": response,
                "repeat_idx": q_data.get("repeat_idx", 0),
            }
            responses.append(response_data)

            # 各層のノルムを記録
            for layer_idx in range(num_layers):
                if layer_idx in all_batch_norms and i < len(all_batch_norms[layer_idx]):
                    all_norms[layer_idx].append(all_batch_norms[layer_idx][i])
                else:
                    all_norms[layer_idx].append(None)

            # 保存対象層の隠れ状態を記録 (事前計算した平均ベクトルを使用)
            for layer_idx in layers_to_save:
                if layer_idx in all_avg_hidden_states and i < len(
                    all_avg_hidden_states[layer_idx]
                ):
                    hidden_states_to_save[layer_idx].append(
                        all_avg_hidden_states[layer_idx][i]
                    )

            # 各層のPersona Vector投影値を記録 (新しいメソッドから)
            for layer_idx in range(num_layers):
                if layer_idx in all_batch_projections and i < len(
                    all_batch_projections[layer_idx]
                ):
                    all_projections[layer_idx].append(
                        all_batch_projections[layer_idx][i]
                    )
                else:
                    all_projections[layer_idx].append(None)

        # メモリクリーンアップ
        tracker.remove_hooks()
        del tracker
        gc.collect()

    return responses, hidden_states_to_save, all_norms, all_projections


def score_responses_batch_local(
    responses: List[Dict],
    trait: str,
    judge_model: str = "openai/gpt-oss-20b",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> pd.DataFrame:
    """ローカルLLMでバッチスコア計算"""
    trait_data = json.load(open(f"data_generation/trait_data_eval/{trait}.json", "r"))

    judge_prompts = {
        trait: trait_data["eval_prompt"],
        "coherence": Prompts.get("coherence_0_100"),
    }

    batch_judge = BatchLocalLLMJudge(
        model_path=judge_model,
        prompt_templates=judge_prompts,
        eval_types={trait: "0_100", "coherence": "0_100"},
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    batch_judge.load_model()

    df_data = []
    for resp in responses:
        df_data.append(
            {
                "question": resp["question"],
                "prompt": resp["prompt"],
                "answer": resp["answer"],
                "question_id": resp["question_id"],
            }
        )

    df = pd.DataFrame(df_data)

    for metric in judge_prompts.keys():
        judge_kwargs_list = [
            {"question": resp["question"], "answer": resp["answer"]}
            for resp in responses
        ]
        scores = batch_judge.batch_judge(metric, judge_kwargs_list)
        df[metric] = scores

    batch_judge.unload_model()

    return df


async def score_responses_batch_openai(
    responses: List[Dict],
    trait: str,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    max_concurrent_judges: int = 10,
) -> pd.DataFrame:
    """OpenAI APIでバッチスコア計算（セマフォア制御）"""
    trait_data = json.load(open(f"data_generation/trait_data_eval/{trait}.json", "r"))

    judges = {
        trait: OpenAiJudge(judge_model, trait_data["eval_prompt"], eval_type="0_100"),
        "coherence": OpenAiJudge(
            judge_model, Prompts.get("coherence_0_100"), eval_type="0_100"
        ),
    }

    df_data = []
    for resp in responses:
        df_data.append(
            {
                "question": resp["question"],
                "prompt": resp["prompt"],
                "answer": resp["answer"],
                "question_id": resp["question_id"],
            }
        )

    df = pd.DataFrame(df_data)

    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        """セマフォアで制御された判定実行関数"""
        async with semaphore:
            await asyncio.sleep(0.1)
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    for metric, judge in judges.items():
        print(
            f"\nScoring {metric} with OpenAI API (max_concurrent={max_concurrent_judges})..."
        )

        all_results = [None] * len(responses)

        tasks = [
            run_with_semaphore(task_idx, judge, resp["question"], resp["answer"])
            for task_idx, resp in enumerate(responses)
        ]

        with tqdm(total=len(tasks), desc=f"Scoring {metric}") as pbar:
            for task in asyncio.as_completed(tasks):
                task_idx, result = await task
                all_results[task_idx] = result
                pbar.update(1)

        df[metric] = all_results

    return df


def save_jsonl_data(
    hidden_states_to_save: Dict[int, List],
    all_norms: Dict[int, List],
    all_projections: Dict[int, List],
    output_dir: str,
):
    """隠れ状態、ノルム、投影値をJSONLで保存"""
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "hidden_states.jsonl")

    # データが空でないことを確認（サンプル数を最初の有効なデータから取得）
    num_samples = 0
    if hidden_states_to_save:
        for states in hidden_states_to_save.values():
            if states:
                num_samples = len(states)
                break
    if num_samples == 0 and all_norms:
        for norms in all_norms.values():
            if norms:
                num_samples = len(norms)
                break

    if num_samples == 0:
        print("No data to save to JSONL.")
        return

    with open(jsonl_path, "w") as f:
        for i in range(num_samples):
            entry = {
                "sample_idx": i,
                "hidden_states": {},
                "norms": {},
                "projections": {},
            }

            for layer_idx, states in hidden_states_to_save.items():
                if i < len(states):
                    if torch.is_tensor(states[i]):
                        tensor_data = states[i].float().cpu().numpy().tolist()
                        entry["hidden_states"][str(layer_idx)] = tensor_data
                    else:
                        entry["hidden_states"][str(layer_idx)] = states[i]

            for layer_idx, norms in all_norms.items():
                if i < len(norms):
                    entry["norms"][str(layer_idx)] = norms[i]

            for layer_idx, projs in all_projections.items():
                if i < len(projs):
                    entry["projections"][str(layer_idx)] = projs[i]

            f.write(json.dumps(entry) + "\n")

    print(f"Saved hidden states, norms, and projections to: {jsonl_path}")


def save_scored_csv(
    responses_df: pd.DataFrame,
    output_dir: str,
):
    """スコア付きの最終的なCSVファイルを保存"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "responses.csv")
    responses_df.to_csv(csv_path, index=False)
    print(f"Saved final scored responses to: {csv_path}")


def save_intermediate_responses(responses: List[Dict], intermediate_path: str):
    """Phase 1後の中間結果を保存"""
    print(f"\nSaving intermediate responses to: {intermediate_path}")

    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    df_intermediate = pd.DataFrame(responses)
    df_intermediate.to_csv(intermediate_path, index=False)
    print(f"Intermediate responses saved: {len(df_intermediate)} rows\n")


def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    trait: str = "evil",
    coef: float = 0.0,
    steering_layer: Optional[int] = None,
    output_base_dir: str = "data/steering_analysis_results",
    judge_model: Optional[str] = None,
    use_openai: bool = False,
    batch_size: int = 100,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    n_per_question: int = 10,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_concurrent_judges: int = 10,
    overwrite: bool = False,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    return_to_original_norm: bool = False,
):
    """メイン実行関数"""
    if judge_model is None:
        judge_model = "openai/gpt-oss-20b" if use_openai else model

    # ステアリング層のデフォルト設定
    if steering_layer is None:
        num_layers = get_model_layer_count(model)
        steering_layer = num_layers // 2
    else:
        num_layers = get_model_layer_count(model)

    org_name = model.split("/")[0]
    model_name = model.split("/")[1]
    output_dir = os.path.join(
        output_base_dir, org_name, model_name, trait, f"coef_{coef}"
    )

    csv_path = os.path.join(output_dir, "responses.csv")
    intermediate_path = get_intermediate_response_path(csv_path)
    layers_to_save = get_selective_layers_to_save(num_layers, steering_layer)

    if os.path.exists(csv_path) and not overwrite:
        print(f"Output already exists: {csv_path}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating {trait} with {model}")
    print(f"Steering coefficient: {coef} at layer {steering_layer}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Phase 1: 質問読み込み・推論・隠れ状態記録
    if not skip_phase1:
        print("\n" + "=" * 60)
        print("Phase 1: Generating Responses with Steering")
        print("=" * 60)

        print("Loading evaluation questions...")
        questions_data = load_eval_questions(trait, version="eval")
        print(f"Loaded {len(questions_data)} questions (pos only)")

        print("Loading Persona Vectors...")
        steering_vectors, _ = load_persona_vectors(trait, model)

        print("Loading model...")
        llm, tokenizer = load_model(model)

        print("Generating responses with hidden state tracking...")
        responses, hidden_states_to_save, all_norms, all_projections = (
            generate_responses_with_projections_and_hidden_states(
                llm,
                tokenizer,
                questions_data,
                coef=coef,
                steering_vectors=steering_vectors,
                steering_layer=steering_layer,
                num_layers=num_layers,
                layers_to_save=layers_to_save,
                batch_size=batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                n_per_question=n_per_question,
                return_to_original_norm=return_to_original_norm,
            )
        )

        print(f"Generated {len(responses)} responses")

        # 中間結果を保存
        save_intermediate_responses(responses, intermediate_path)
        save_jsonl_data(hidden_states_to_save, all_norms, all_projections, output_dir)

        # メモリ解放
        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    else:
        # Phase 1をスキップ：中間ファイルから読み込み
        print("\n" + "=" * 60)
        print("Phase 1: Skipped (loading intermediate responses)")
        print("=" * 60)

        df_intermediate = pd.read_csv(intermediate_path)
        responses = df_intermediate.to_dict("records")

        layers_to_save = get_selective_layers_to_save(num_layers, steering_layer)
        hidden_states_to_save = {layer: [] for layer in layers_to_save}
        all_norms = {i: [] for i in range(num_layers)}
        all_projections = {i: [] for i in range(num_layers)}

        print(f"Loaded {len(responses)} responses from intermediate file")

    # Phase 2: スコア計算
    if not skip_phase2:
        print("\n" + "=" * 60)
        print("Phase 2: Scoring Responses")
        print("=" * 60)

        if use_openai:
            print("Scoring with OpenAI API...")
            responses_df = asyncio.run(
                score_responses_batch_openai(
                    responses,
                    trait,
                    judge_model=judge_model,
                    max_concurrent_judges=max_concurrent_judges,
                )
            )
        else:
            print("Scoring with local LLM...")
            responses_df = score_responses_batch_local(
                responses,
                trait,
                judge_model=judge_model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        # Phase 3: 結果を保存
        print("\n" + "=" * 60)
        print("Phase 3: Saving Results")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        save_scored_csv(responses_df, output_dir)

        # 中間ファイルを削除
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)
            print(f"Deleted intermediate file: {intermediate_path}")

        # サマリー表示
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        for col in [trait, "coherence"]:
            if col in responses_df.columns:
                print(
                    f"{col}: {responses_df[col].mean():.2f} ± {responses_df[col].std():.2f}"
                )
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
