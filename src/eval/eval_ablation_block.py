"""
eval_persona_ablation.py

指定した層のattn/mlp出力からPersona Vector方向の成分を除去（アブレーション）して評価を行うスクリプト

Ablationとは：出力ベクトルからPersona Vectorの方向成分を除去すること
ablated_output = output - (output · unit_persona_vector) * unit_persona_vector
"""

import asyncio
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm, trange

from eval.model_utils import load_model, load_vllm_model
from src.activation_steer.activation_ablation_block import ActivationAblator
from src.chat_template_utils import apply_chat_template_safe
from src.eval.common.loaders import load_persona_questions
from src.eval.common.question import Question


def sample_with_ablation(
    model: object,
    tokenizer: object,
    conversations: list[list[dict[str, str]]],
    persona_vector: torch.Tensor,
    layer: int,
    ablation_type: str = "attn_output",
    positions: str = "all",
    bs: int = 100,
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
):
    """アブレーションを適用してモデルの応答を生成する

    Args:
        model: 使用する言語モデル
        tokenizer: トークナイザー
        conversations (list): 会話データのリスト
        persona_vector (torch.Tensor): Persona Vector（この方向成分を除去）
        layer (int): 適用するレイヤー番号（0-based index、0=1層目）
        ablation_type (str): アブレーションタイプ（"attn_output"|"mlp_output"）
        positions (str): 適用位置（"all"|"prompt"|"response"）
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
                        f"Warning: System role not supported by model, converting system messages to user role"
                    )

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating with ablation"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        # ActivationAblatorを設定（Persona Vector方向成分を除去）
        ablator = ActivationAblator(
            model,
            persona_vector=persona_vector,
            layer_idx=layer,
            ablation_type=ablation_type,
            positions=positions,
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
    model: object,
    tokenizer: object,
    conversations: list[list[dict[str, str]]],
    bs: int = 100,
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
):
    """アブレーションなしでモデルの応答を生成する（ベースライン用）

    Args:
        model: 使用する言語モデル
        tokenizer: トークナイザー
        conversations (list): 会話データのリスト
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


async def eval_batched(
    questions: list[Question],
    llm: object,
    tokenizer: object,
    persona_vector: torch.Tensor | None = None,
    layer: int | None = None,
    ablation_type: str = "attn_output",
    positions: str = "all",
    batch_size: int = 100,
    n_per_question: int = 100,
    max_concurrent_judges: int = 4,
    max_tokens: int = 1000,
):
    """すべての質問を一括処理して高速な推論を実現する

    Args:
        questions (list): Questionオブジェクトのリスト
        llm: 使用する言語モデル
        tokenizer: トークナイザー
        persona_vector (torch.Tensor): Persona Vector（この方向成分を除去）
        layer (int): アブレーションするレイヤー（Noneの場合はアブレーションなし）
        ablation_type (str): アブレーションタイプ（"attn_output"|"mlp_output"）
        positions (str): 適用位置（"all"|"prompt"|"response"）
        batch_size (int): バッチサイズ（デフォルト: 100）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 100）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 4）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）

    Returns:
        list: 各質問の評価結果を含むDataFrameのリスト
    """
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

    if layer is not None and persona_vector is not None:
        prompts, answers = sample_with_ablation(
            llm,
            tokenizer,
            all_conversations,
            persona_vector=persona_vector,
            layer=layer,
            ablation_type=ablation_type,
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
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        # Create dataframe for this question
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

        # Collect all judge tasks
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

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            await asyncio.sleep(0.1)
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    # Create all tasks with semaphore control
    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    # Process tasks in batches with progress bar
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            pbar.update(1)

    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model: str,
    trait: str,
    output_path: str,
    vector_path: str | None = None,
    layer: int | None = None,
    ablation_type: str = "attn_output",
    positions: str = "all",
    max_tokens: int = 1000,
    n_per_question: int = 5,
    batch_size: int = 100,
    max_concurrent_judges: int = 4,
    persona_instruction_type: str | None = None,
    assistant_name: str | None = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    version: str = "extract",
    overwrite: bool = False,
):
    """評価を実行する

    Args:
        model (str): 評価するモデル名
        trait (str): 評価する特性名
        output_path (str): 結果を保存するCSVファイルパス
        vector_path (str): Persona Vectorファイルのパス（アブレーションに使用）
        layer (int): アブレーションするレイヤー（Noneの場合はアブレーションなし）
        ablation_type (str): アブレーションタイプ（"attn_output"|"mlp_output"）
        positions (str): 適用位置（"all"|"prompt"|"response"）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 5）
        batch_size (int): バッチサイズ（デフォルト: 100）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 4）
        persona_instruction_type (str): ペルソナ指示タイプ（オプション）
        assistant_name (str): アシスタント名（オプション）
        judge_model (str): 判定用モデル名（デフォルト: "gpt-4.1-mini-2025-04-14"）
        version (str): データバージョン（デフォルト: "extract"）
        overwrite (bool): 既存ファイルを上書きするか（デフォルト: False）
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        print(
            f"coherence:  {df['coherence'].mean():.2f} +- {df['coherence'].std():.2f}"
        )
        return

    print(f"Output path: {output_path}")
    print(f"Layer: {layer}, Ablation type: {ablation_type}, Positions: {positions}")
    if vector_path:
        print(f"Vector path: {vector_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0

    # Load model and vector
    persona_vector = None
    if layer is not None and vector_path is not None:
        # アブレーションを使用する場合はHuggingFaceモデルを使用
        llm, tokenizer = load_model(model)
        # ベクトルファイルのindex layerを取得（0=1層目、1=2層目、...）
        all_vectors = torch.load(vector_path, weights_only=False)
        persona_vector = all_vectors[layer]
        print(f"Loaded persona vector for layer {layer}, shape: {persona_vector.shape}")
    else:
        # アブレーションなしの場合はvLLMモデルを使用可能
        llm, tokenizer = load_vllm_model(model)

    # Load questions
    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version,
    )

    print(f"Batch processing {len(questions)} '{trait}' questions...")
    outputs_list = asyncio.run(
        eval_batched(
            questions,
            llm,
            tokenizer,
            persona_vector=persona_vector,
            layer=layer,
            ablation_type=ablation_type,
            positions=positions,
            batch_size=batch_size,
            n_per_question=n_per_question,
            max_concurrent_judges=max_concurrent_judges,
            max_tokens=max_tokens,
        )
    )
    outputs = pd.concat(outputs_list)

    outputs.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"{trait}:  {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")
    print(
        f"coherence:  {outputs['coherence'].mean():.2f} +- {outputs['coherence'].std():.2f}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
