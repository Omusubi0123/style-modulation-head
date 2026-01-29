"""
eval_style_head_ablation.py - Style HeadをZero Ablationして評価

Style Headを累積的にZero Ablationしていき、
生成される文章がどうなるか（無機質になるのか、system promptが効かなくなるのか）を調べる。

実験フロー:
1. Style Head CSVを読み込む
2. 1行目のヘッドをアブレーション → 生成 → スコア計測
3. 1行目+2行目のヘッドをアブレーション → 生成 → スコア計測
4. 累積的にアブレーション範囲を広げていく

使い方:
    python -m eval.eval_style_head_ablation \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait agreeableness \
        --output_dir results/style_head_ablation
"""

import asyncio
import json
import logging
import os
import random

import pandas as pd
import torch
from tqdm import tqdm, trange

from eval.model_utils import load_model
from eval.prompts import Prompts
from src.activation_steer.activation_ablator_head import (
    ActivationAblatorHeadMultiple,
    load_style_heads_from_csv,
)
from src.chat_template_utils import apply_chat_template_safe
from src.config import setup_credentials
from src.eval.common.openai_judge import OpenAiJudge

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()


def get_model_short_name(model_name: str) -> str:
    """モデル名からCSVファイル名用の短縮名を取得する

    例:
    - "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b-instruct"
    - "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b-instruct"
    """
    short_name = model_name.split("/")[-1].lower()
    return short_name


def sample_with_ablation(
    model,
    tokenizer,
    conversations,
    ablation_instructions: list[dict],
    positions: str = "response",
    bs: int = 100,
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
):
    """複数層の複数ヘッドをZero Ablationして応答を生成する

    Args:
        model: 使用する言語モデル
        tokenizer: トークナイザー
        conversations (list): 会話データのリスト
        ablation_instructions (list[dict]): アブレーション指示のリスト
            各dictは {"layer_idx": int, "head_indices": List[int], "positions": str}
        positions (str): アブレーション適用位置（デフォルト: "response"）
        bs (int): バッチサイズ（デフォルト: 100）
        top_p (float): nucleus samplingパラメータ（デフォルト: 1）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        temperature (float): サンプリング温度（デフォルト: 1）
        min_tokens (int): 最小生成トークン数（デフォルト: 1）

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    # トークナイザーの設定
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # プロンプトを準備
    prompts = []
    for i, messages in enumerate(conversations):
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        if i == 0 and any(msg["role"] == "system" for msg in messages):
            try:
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                if "System role not supported" in str(e) or "system" in str(e).lower():
                    print(
                        "Warning: System role not supported by model, "
                        "converting system messages to user role"
                    )

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating with ablation"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        # 各指示にpositionsを追加
        instructions_with_positions = []
        for inst in ablation_instructions:
            instructions_with_positions.append({
                "layer_idx": inst["layer_idx"],
                "head_indices": inst["head_indices"],
                "positions": inst.get("positions", positions),
            })

        # ActivationAblatorHeadMultipleを設定
        ablator = ActivationAblatorHeadMultiple(
            model,
            instructions_with_positions,
            debug=False,
        )

        with ablator:
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(batch_outputs)

    return prompts, outputs


def sample_without_ablation(
    model,
    tokenizer,
    conversations,
    bs: int = 100,
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
):
    """アブレーションなしで応答を生成する（ベースライン用）"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, messages in enumerate(conversations):
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating without ablation"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        with torch.no_grad():
            output = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                min_new_tokens=min_tokens,
            )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(batch_outputs)

    return prompts, outputs


def load_jsonl(path):
    """JSONLファイルを読み込んで辞書のリストとして返す"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question:
    """評価用の質問クラス"""

    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4.1-mini-2025-04-14",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(
                judge,
                prompt,
                eval_type=judge_eval_type if metric != "coherence" else "0_100",
            )
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations


def a_or_an(word):
    """英語の不定冠詞を決定するヘルパー関数"""
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="eval",
):
    """ペルソナ評価用の質問リストを読み込み、Questionオブジェクトのリストを作成する"""
    trait_data = json.load(
        open(f"data_generation/trait_data_{version}/{trait}.json", "r")
    )
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [
                x[persona_instructions_type] for x in trait_data["instruction"]
            ]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name_to_use = trait
                    else:
                        assistant_name_to_use = "helpful"
                else:
                    assistant_name_to_use = assistant_name
                system = f"You are {a_or_an(assistant_name_to_use)} {assistant_name_to_use} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type,
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type,
                )
            )
    return questions


async def eval_batched(
    questions,
    llm,
    tokenizer,
    ablation_instructions: list[dict] = None,
    positions: str = "response",
    batch_size: int = 100,
    n_per_question: int = 100,
    max_concurrent_judges: int = 4,
    max_tokens: int = 1000,
):
    """すべての質問を一括処理して高速な推論を実現する"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")

    if ablation_instructions:
        prompts, answers = sample_with_ablation(
            llm,
            tokenizer,
            all_conversations,
            ablation_instructions=ablation_instructions,
            positions=positions,
            bs=batch_size,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
        )
    else:
        prompts, answers = sample_without_ablation(
            llm,
            tokenizer,
            all_conversations,
            bs=batch_size,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
        )

    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        df_data = [
            dict(
                question=question_text,
                prompt=prompt,
                answer=answer,
                question_id=question.id,
            )
            for question_text, answer, prompt in zip(
                q_paraphrases, q_answers, q_prompts
            )
        ]

        df = pd.DataFrame(df_data)
        question_dfs.append(df)

        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(
                zip(q_paraphrases, q_answers)
            ):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    # Run judge evaluations with concurrency control
    print(
        f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests..."
    )
    all_results = [None] * len(all_judge_tasks)

    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            await asyncio.sleep(0.1)
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            pbar.update(1)

    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def format_ablation_info(ablation_instructions: list[dict]) -> str:
    """アブレーション情報をファイル名用に整形する"""
    parts = []
    for inst in ablation_instructions:
        layer = inst["layer_idx"]
        heads = "-".join(str(h) for h in inst["head_indices"])
        parts.append(f"L{layer}H{heads}")
    return "_".join(parts)


def main(
    model: str,
    trait: str,
    output_dir: str,
    style_head_csv: str = None,
    positions: str = "response",
    max_tokens: int = 1000,
    n_per_question: int = 5,
    batch_size: int = 100,
    max_concurrent_judges: int = 4,
    persona_instruction_type: str = None,
    assistant_name: str = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    version: str = "extract",
    overwrite: bool = False,
):
    """累積的にStyle Headをアブレーションして評価を実行する

    Args:
        model (str): 評価するモデル名
        trait (str): 評価する特性名
        output_dir (str): 結果を保存するディレクトリ
        style_head_csv (str): Style Head CSVファイルのパス（Noneの場合は自動検出）
        positions (str): アブレーション適用位置（デフォルト: "response"）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 5）
        batch_size (int): バッチサイズ（デフォルト: 100）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 4）
        persona_instruction_type (str): ペルソナ指示タイプ（オプション）
        assistant_name (str): アシスタント名（オプション）
        judge_model (str): 判定用モデル名
        version (str): データバージョン（デフォルト: "extract"）
        overwrite (bool): 既存ファイルを上書きするか（デフォルト: False）
    """
    # Style Head CSVのパスを決定
    if style_head_csv is None:
        model_short_name = get_model_short_name(model)
        style_head_csv = f"style_head/{model_short_name}.csv"

    print(f"Model: {model}")
    print(f"Trait: {trait}")
    print(f"Style Head CSV: {style_head_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Positions: {positions}")

    # Style Head情報を読み込む
    style_heads = load_style_heads_from_csv(style_head_csv)
    print(f"Loaded {len(style_heads)} style head entries")
    for i, sh in enumerate(style_heads):
        print(f"  Entry {i+1}: layer={sh['layer']+1} (1-indexed), cor_heads={[h+1 for h in sh['cor_heads']]} (1-indexed)")

    os.makedirs(output_dir, exist_ok=True)

    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0

    # モデルをロード
    llm, tokenizer = load_model(model)

    # 質問をロード
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version,
    )

    # サマリー用のリスト
    summary_results = []

    # ベースライン（アブレーションなし）の評価
    baseline_output_path = os.path.join(
        output_dir,
        f"{trait}_baseline_{persona_instruction_type or 'none'}.csv"
    )

    if os.path.exists(baseline_output_path) and not overwrite:
        print(f"Baseline already exists: {baseline_output_path}")
        baseline_df = pd.read_csv(baseline_output_path)
    else:
        print("=== Evaluating baseline (no ablation) ===")
        baseline_outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                ablation_instructions=None,
                positions=positions,
                batch_size=batch_size,
                n_per_question=n_per_question,
                max_concurrent_judges=max_concurrent_judges,
                max_tokens=max_tokens,
            )
        )
        baseline_df = pd.concat(baseline_outputs_list)
        baseline_df["ablation_step"] = 0
        baseline_df["ablated_layers"] = ""
        baseline_df.to_csv(baseline_output_path, index=False)
        print(f"Saved baseline to: {baseline_output_path}")

    print(f"Baseline - {trait}: {baseline_df[trait].mean():.2f} +- {baseline_df[trait].std():.2f}")
    print(f"Baseline - coherence: {baseline_df['coherence'].mean():.2f} +- {baseline_df['coherence'].std():.2f}")

    summary_results.append({
        "step": 0,
        "ablated_layers": "",
        "ablated_heads": "",
        f"{trait}_mean": baseline_df[trait].mean(),
        f"{trait}_std": baseline_df[trait].std(),
        "coherence_mean": baseline_df["coherence"].mean(),
        "coherence_std": baseline_df["coherence"].std(),
    })

    # 累積的にアブレーション
    cumulative_instructions = []

    for step, style_head_entry in enumerate(style_heads, start=1):
        layer_idx = style_head_entry["layer"]
        head_indices = style_head_entry["cor_heads"]

        if not head_indices:
            print(f"Step {step}: No cor_heads for layer {layer_idx+1}, skipping...")
            continue

        # 累積的に追加
        cumulative_instructions.append({
            "layer_idx": layer_idx,
            "head_indices": head_indices,
        })

        # ファイル名を生成
        ablated_layers_str = "_".join(
            f"L{inst['layer_idx']+1}" for inst in cumulative_instructions
        )
        output_path = os.path.join(
            output_dir,
            f"{trait}_ablation_step{step}_{ablated_layers_str}_{persona_instruction_type or 'none'}.csv"
        )

        if os.path.exists(output_path) and not overwrite:
            print(f"Step {step} already exists: {output_path}")
            step_df = pd.read_csv(output_path)
        else:
            print(f"=== Step {step}: Ablating cumulative layers {ablated_layers_str} ===")
            print(f"  Adding layer {layer_idx+1} (0-indexed: {layer_idx}), heads {[h+1 for h in head_indices]} (0-indexed: {head_indices})")
            print(f"  Total instructions: {len(cumulative_instructions)}")

            step_outputs_list = asyncio.run(
                eval_batched(
                    questions,
                    llm,
                    tokenizer,
                    ablation_instructions=cumulative_instructions,
                    positions=positions,
                    batch_size=batch_size,
                    n_per_question=n_per_question,
                    max_concurrent_judges=max_concurrent_judges,
                    max_tokens=max_tokens,
                )
            )
            step_df = pd.concat(step_outputs_list)
            step_df["ablation_step"] = step
            step_df["ablated_layers"] = ablated_layers_str
            step_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")

        trait_mean = step_df[trait].mean()
        trait_std = step_df[trait].std()
        coherence_mean = step_df["coherence"].mean()
        coherence_std = step_df["coherence"].std()

        print(f"Step {step} - {trait}: {trait_mean:.2f} +- {trait_std:.2f}")
        print(f"Step {step} - coherence: {coherence_mean:.2f} +- {coherence_std:.2f}")

        # ヘッド情報を整形
        ablated_heads_str = "; ".join(
            f"L{inst['layer_idx']+1}:{[h+1 for h in inst['head_indices']]}"
            for inst in cumulative_instructions
        )

        summary_results.append({
            "step": step,
            "ablated_layers": ablated_layers_str,
            "ablated_heads": ablated_heads_str,
            f"{trait}_mean": trait_mean,
            f"{trait}_std": trait_std,
            "coherence_mean": coherence_mean,
            "coherence_std": coherence_std,
        })

    # サマリーを保存
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(
        output_dir,
        f"{trait}_ablation_summary_{persona_instruction_type or 'none'}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print("\n=== Final Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

