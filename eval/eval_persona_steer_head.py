"""
eval_persona_steer_head.py - ヘッド単位のsteeringの評価

特定の層の特定のヘッドのみをsteeringして、
そのヘッドがステアリング効果が大きいかどうかを確認する。

使い方:
    python -m eval.eval_persona_steer_head \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait agreeableness \
        --output_path results/head_steering.csv \
        --vector_path vectors/agreeableness_prompt_avg_diff_attn_pre_o_proj.pt \
        --layer 15 \
        --head_indices 0 5 10 \
        --coef 1.5
"""

import asyncio
import json
import logging
import os
import random

import pandas as pd
import torch
from tqdm import tqdm

from eval.model_utils import load_model, load_vllm_model
from eval.prompts import Prompts
from src.config import setup_credentials
from src.judge import OpenAiJudge
from src.visualize.projection_tracker import ProjectionTrackerHead

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()


def sample_steering(
    model,
    tokenizer,
    conversations,
    vector,
    layer,
    head_indices,
    coef,
    bs=100,
    top_p=1,
    max_tokens=1000,
    temperature=1,
    min_tokens=1,
    steering_type="response",
):
    """ヘッド単位のステアリングベクトルを使用してモデルの応答を生成する

    Args:
        model: 使用する言語モデル
        tokenizer: トークナイザー
        conversations (list): 会話データのリスト
        vector: ステアリングベクトル
        layer (int): 適用するレイヤー番号（0-based index、0=1層目）
        head_indices (list): ステアリング対象のヘッドインデックスのリスト
        coef (float): ステアリング係数
        bs (int): バッチサイズ（デフォルト: 100）
        top_p (float): nucleus samplingパラメータ（デフォルト: 1）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        temperature (float): サンプリング温度（デフォルト: 1）
        min_tokens (int): 最小生成トークン数（デフォルト: 1）
        steering_type (str): ステアリングタイプ（デフォルト: "response"）

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    tracker = ProjectionTrackerHead(
        model, tokenizer, vector, layer, head_indices, steering_type
    )

    prompts, outputs, _ = tracker.generate_with_projections(
        conversations,
        coef=coef,
        bs=bs,
        top_p=top_p,
        max_tokens=max_tokens,
        temperature=temperature,
        min_tokens=min_tokens,
    )

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
        judge: str = "gpt-4o",
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

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        head_indices=None,
        max_tokens=1000,
        n_per_question=100,
        steering_type="last",
    ):
        paraphrases, conversations = self.get_input(n_per_question)

        result = sample_steering(
            llm,
            tokenizer,
            conversations,
            vector,
            layer,
            head_indices,
            coef,
            temperature=self.temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
        prompts, answers = result

        df_data = [
            dict(question=question, prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ]

        df = pd.DataFrame(df_data)
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df


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
                        assistant_name = trait
                    else:
                        assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
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
    coef,
    vector=None,
    layer=None,
    head_indices=None,
    batch_size=100,
    n_per_question=100,
    max_concurrent_judges=4,
    max_tokens=1000,
    steering_type="last",
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

    result = sample_steering(
        llm,
        tokenizer,
        all_conversations,
        vector,
        layer,
        head_indices,
        coef,
        bs=batch_size,
        temperature=questions[0].temperature,
        max_tokens=max_tokens,
        steering_type=steering_type,
    )
    prompts, answers = result

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


def main(
    model,
    trait,
    output_path,
    coef=0,
    vector_path=None,
    layer=None,
    head_indices=None,
    steering_type="response",
    max_tokens=1000,
    n_per_question=5,
    batch_process=True,
    max_concurrent_judges=2,
    batch_size=100,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
    overwrite=False,
):
    """評価を実行する

    Args:
        model (str): 評価するモデル名
        trait (str): 評価する特性名
        output_path (str): 結果を保存するCSVファイルパス
        coef (float): ステアリング係数（デフォルト: 0）
        vector_path (str): ステアリングベクトルのパス（オプション）
        layer (int): ステアリングレイヤー（オプション）
        head_indices (list or str): ステアリング対象のヘッドインデックス（カンマ区切りの文字列または整数リスト）
        steering_type (str): ステアリングタイプ（デフォルト: "response"）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 5）
        batch_process (bool): バッチ処理を使用するか（デフォルト: True）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 4）
        batch_size (int): バッチサイズ（デフォルト: 100）
        persona_instruction_type (str): ペルソナ指示タイプ（オプション）
        assistant_name (str): アシスタント名（オプション）
        judge_model (str): 判定用モデル名
        version (str): データバージョン（デフォルト: "extract"）
        overwrite (bool): 既存ファイルを上書きするか（デフォルト: False）
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for t in [trait, "coherence"]:
            print(f"{t}:  {df[t].mean():.2f} +- {df[t].std():.2f}")
        return

    # head_indicesが文字列の場合はパース
    if isinstance(head_indices, str):
        head_indices = [int(x.strip()) for x in head_indices.split(",")]
    if isinstance(head_indices, int):
        head_indices = [head_indices]
    if isinstance(head_indices, tuple): # Fireがカンマ区切りの文字列をタプルに変換した場合
        head_indices = list(head_indices)

    print(f"Output path: {output_path}")
    print(f"Head indices: {head_indices}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0

    if (
        coef != 0
        and vector_path is not None
        and layer is not None
        and head_indices is not None
    ):
        llm, tokenizer = load_model(model)
        # ベクトルファイルのindex layerを取得（0=1層目、1=2層目、...）
        vector = torch.load(vector_path, weights_only=False)[layer]
        print(f"Loaded vector from {vector_path} layer {layer}")
        print(f"Vector shape: {vector.shape}")
    else:
        llm, tokenizer, _ = load_vllm_model(model)
        vector = None
        head_indices = None

    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version,
    )

    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                coef,
                vector,
                layer,
                head_indices,
                batch_size,
                n_per_question,
                max_concurrent_judges,
                max_tokens,
                steering_type=steering_type,
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        tokenizer,
                        coef,
                        vector,
                        layer,
                        head_indices,
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                    )
                )
            )
        outputs = pd.concat(outputs)

    # 結果にヘッド情報を追加
    if head_indices is not None:
        outputs["head_indices"] = str(head_indices)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    for t in [trait, "coherence"]:
        print(f"{t}:  {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
