"""
loaders.py - データ読み込みユーティリティ
"""

import json
from typing import List

from src.eval.prompts import Prompts
from src.eval.common.question import Question
from src.eval.common.utils import a_or_an


def load_jsonl(path: str) -> List[dict]:
    """JSONLファイルを読み込んで辞書のリストとして返す"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


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

