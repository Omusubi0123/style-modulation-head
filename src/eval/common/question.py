"""
question.py - Question class for evaluation
"""

import random
from typing import List

from src.eval.common.openai_judge import OpenAiJudge


class Question:
    """Question class for evaluation

    Provides functionality to generate model responses for individual
    questions and evaluate them with multiple judges.
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
        """Initialize question class

        Args:
            id: Unique identifier for the question
            paraphrases: List of question paraphrases
            judge_prompts: Dictionary of judge prompts
            temperature: Sampling temperature (default: 1)
            system: System message (optional)
            judge: Judge model name (default: "gpt-4.1-mini-2025-04-14")
            judge_eval_type: Judge type (default: "0_100")
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
        """Generate input data from question

        Args:
            n_per_question: Number of samples per question

        Returns:
            tuple: (list of selected paraphrases, list of conversation formats)
        """
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations
