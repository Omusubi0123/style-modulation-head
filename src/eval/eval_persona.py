"""
eval_persona.py - ペルソナ評価

TransformerブロックレベルのステアリングでPersona評価を行う。
"""

import asyncio
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.eval.common.judge import run_judge_evaluations
from src.eval.common.loaders import load_persona_questions
from src.eval.common.utils import print_results
from src.eval.model_utils import load_model, load_vllm_model
from src.eval.sampling import sample_vllm, sample_with_steering
from src.config import setup_credentials

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

config = setup_credentials()


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    batch_size=100,
    n_per_question=5,
    max_concurrent_judges=4,
    max_tokens=1000,
    steering_type="response",
):
    """すべての質問を一括処理して高速な推論を実現する"""
    # 全プロンプトを収集
    all_paraphrases = []
    all_conversations = []
    question_indices = []

    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    print(f"Generating {len(all_conversations)} responses in a single batch...")

    # 回答を生成
    if coef != 0 and vector is not None and layer is not None:
        prompts, answers = sample_with_steering(
            llm,
            tokenizer,
            all_conversations,
            vector,
            layer,
            coef,
            bs=batch_size,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    else:
        prompts, answers = sample_vllm(
            llm,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
        )

    # ジャッジ評価を実行
    question_dfs = await run_judge_evaluations(
        questions,
        all_paraphrases,
        answers,
        question_indices,
        prompts,
        max_concurrent_judges,
    )

    return question_dfs


def main(
    model,
    trait,
    output_path,
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=1000,
    n_per_question=5,
    batch_process=True,
    max_concurrent_judges=4,
    batch_size=100,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
):
    """評価を実行する

    Args:
        model: 評価するモデル名
        trait: 評価する特性名
        output_path: 結果を保存するCSVファイルパス
        coef: ステアリング係数（デフォルト: 0）
        vector_path: ステアリングベクトルのパス（オプション）
        layer: ステアリングレイヤー（オプション）
        steering_type: ステアリングタイプ（デフォルト: "response"）
        max_tokens: 最大生成トークン数（デフォルト: 1000）
        n_per_question: 質問あたりのサンプル数（デフォルト: 5）
        batch_process: バッチ処理を使用するか（デフォルト: True）
        max_concurrent_judges: 最大同時判定数（デフォルト: 4）
        batch_size: バッチサイズ（デフォルト: 100）
        persona_instruction_type: ペルソナ指示タイプ（オプション）
        assistant_name: アシスタント名（オプション）
        judge_model: 判定用モデル名
        version: データバージョン（デフォルト: "extract"）
    """
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    if coef != 0 and vector_path is not None and layer is not None:
        llm, tokenizer = load_model(model)
        vector = torch.load(vector_path, weights_only=False)[layer]
    else:
        llm, tokenizer, _ = load_vllm_model(model)
        vector = None

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
            paraphrases, conversations = question.get_input(n_per_question)
            
            if coef != 0 and vector is not None and layer is not None:
                prompts, answers = sample_with_steering(
                    llm, tokenizer, conversations, vector, layer, coef,
                    temperature=question.temperature, max_tokens=max_tokens,
                    steering_type=steering_type,
                )
            else:
                prompts, answers = sample_vllm(
                    llm, tokenizer, conversations,
                    temperature=question.temperature, max_tokens=max_tokens,
                )
            
            df_data = [
                dict(question=q, prompt=p, answer=a, question_id=question.id)
                for q, a, p in zip(paraphrases, answers, prompts)
            ]
            df = pd.DataFrame(df_data)
            
            for score, judge in question.judges.items():
                scores = asyncio.run(asyncio.gather(*[
                    judge(question=q, answer=a)
                    for q, a in zip(paraphrases, answers)
                ]))
                df[score] = scores
            
            outputs.append(df)
        outputs = pd.concat(outputs)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    print_results(outputs, trait)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
