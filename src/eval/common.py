"""
common.py - 評価用の共通ユーティリティ

eval_persona*.pyファイルで共通に使用する機能を提供する。
"""

import asyncio
import json
import random
from typing import List

import pandas as pd
from tqdm import tqdm

from eval.prompts import Prompts
from src.judge import OpenAiJudge


def load_jsonl(path: str) -> list[dict]:
    """JSONLファイルを読み込んで辞書のリストとして返す"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def a_or_an(word: str) -> str:
    """英語の不定冠詞を決定するヘルパー関数"""
    return "an" if word[0].lower() in "aeiou" else "a"


class Question:
    """評価用の質問クラス

    個々の質問に対してモデルの応答を生成し、
    複数の判定者で評価を行う機能を提供する。
    """

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
        """質問クラスの初期化

        Args:
            id: 質問の一意識別子
            paraphrases: 質問の言い換えリスト
            judge_prompts: 判定用プロンプトの辞書
            temperature: サンプリング温度（デフォルト: 1）
            system: システムメッセージ（オプション）
            judge: 判定用モデル名（デフォルト: "gpt-4.1-mini-2025-04-14"）
            judge_eval_type: 判定タイプ（デフォルト: "0_100"）
        """
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

    def get_input(self, n_per_question: int):
        """質問から入力データを生成する

        Args:
            n_per_question: 質問あたりのサンプル数

        Returns:
            tuple: (選択された言い換えリスト, 会話フォーマットのリスト)
        """
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations


def load_persona_questions(
    trait: str,
    temperature: float = 1,
    persona_instructions_type: str = None,
    assistant_name: str = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    eval_type: str = "0_100",
    version: str = "eval",
) -> List[Question]:
    """ペルソナ評価用の質問リストを読み込み、Questionオブジェクトのリストを作成する

    Args:
        trait: 評価する特性名
        temperature: サンプリング温度（デフォルト: 1）
        persona_instructions_type: ペルソナ指示タイプ（"pos", "neg", またはNone）
        assistant_name: アシスタント名（オプション）
        judge_model: 判定用モデル名（デフォルト: "gpt-4.1-mini-2025-04-14"）
        eval_type: 評価タイプ（デフォルト: "0_100"）
        version: データバージョン（デフォルト: "eval"）

    Returns:
        Questionオブジェクトのリスト
    """
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
                name = assistant_name
                if name is None:
                    name = trait if persona_instructions_type == "pos" else "helpful"
                system = f"You are {a_or_an(name)} {name} assistant. {instruction}"
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


async def run_judge_evaluations(
    questions: List[Question],
    all_paraphrases: List[str],
    all_answers: List[str],
    question_indices: List[int],
    prompts: List[str],
    max_concurrent_judges: int = 4,
) -> List[pd.DataFrame]:
    """すべての質問に対してジャッジ評価を実行する

    Args:
        questions: Questionオブジェクトのリスト
        all_paraphrases: すべての質問テキストのリスト
        all_answers: すべての回答のリスト
        question_indices: 各回答が属する質問のインデックス
        prompts: プロンプトのリスト
        max_concurrent_judges: 最大同時判定数

    Returns:
        各質問の評価結果を含むDataFrameのリスト
    """
    # 質問ごとのDataFrameを準備
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # (question_idx, metric, sample_idx)

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [all_answers[j] for j in indices]

        df_data = [
            dict(
                question=question_text,
                prompt=prompt,
                answer=answer,
                question_id=question.id,
            )
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ]

        df = pd.DataFrame(df_data)
        question_dfs.append(df)

        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(
                zip(q_paraphrases, q_answers)
            ):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    # ジャッジ評価を実行
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

    # 結果をDataFrameに反映
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def print_results(outputs: pd.DataFrame, trait: str) -> None:
    """評価結果を表示する

    Args:
        outputs: 評価結果のDataFrame
        trait: 評価した特性名
    """
    for t in [trait, "coherence"]:
        print(f"{t}:  {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")

