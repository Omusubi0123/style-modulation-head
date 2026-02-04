"""
judge.py - Judge evaluation functionality
"""

import asyncio
from typing import List

import pandas as pd
from tqdm import tqdm

from src.eval.common.question import Question


async def run_judge_evaluations(
    questions: List[Question],
    all_paraphrases: List[str],
    all_answers: List[str],
    question_indices: List[int],
    prompts: List[str],
    max_concurrent_judges: int = 4,
) -> List[pd.DataFrame]:
    """Run judge evaluations for all questions

    Args:
        questions: List of Question objects
        all_paraphrases: List of all question texts
        all_answers: List of all answers
        question_indices: Index of question each answer belongs to
        prompts: List of prompts
        max_concurrent_judges: Maximum concurrent judge calls

    Returns:
        List of DataFrames containing evaluation results for each question
    """
    # Prepare DataFrames for each question
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

    # Run judge evaluations
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

    # Apply results to DataFrames
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs
