"""
question.py - 評価用の質問クラス
"""

import random
from typing import List

from src.eval.common.openai_judge import OpenAiJudge


class Question:
    """評価用の質問クラス

    個々の質問に対してモデルの応答を生成し、
    複数の判定者で評価を行う機能を提供する。
    """

    def __init__(
        self,
        id: str,
        paraphrases: List[str],
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

