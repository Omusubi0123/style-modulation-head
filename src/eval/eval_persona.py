"""
eval_persona.py - Persona evaluation

Perform persona evaluation using transformer block level steering.
"""

import asyncio
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.config import setup_credentials
from src.eval.common.judge import run_judge_evaluations
from src.eval.common.loaders import load_persona_questions
from src.eval.common.utils import print_results
from src.eval.model_utils import load_model, load_vllm_model
from src.eval.sampling import sample_vllm, sample_with_steering

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
    """Process all questions in batch for fast inference"""
    # Collect all prompts
    all_paraphrases = []
    all_conversations = []
    question_indices = []

    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    print(f"Generating {len(all_conversations)} responses in a single batch...")

    # Generate answers
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

    # Run judge evaluations
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
    """Execute evaluation

    Args:
        model: Model name to evaluate
        trait: Trait name to evaluate
        output_path: CSV file path to save results
        coef: Steering coefficient (default: 0)
        vector_path: Path to steering vector (optional)
        layer: Steering layer (optional)
        steering_type: Steering type (default: "response")
        max_tokens: Maximum generation tokens (default: 1000)
        n_per_question: Samples per question (default: 5)
        batch_process: Use batch processing (default: True)
        max_concurrent_judges: Maximum concurrent judges (default: 4)
        batch_size: Batch size (default: 100)
        persona_instruction_type: Persona instruction type (optional)
        assistant_name: Assistant name (optional)
        judge_model: Judge model name
        version: Data version (default: "extract")
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
                    llm,
                    tokenizer,
                    conversations,
                    vector,
                    layer,
                    coef,
                    temperature=question.temperature,
                    max_tokens=max_tokens,
                    steering_type=steering_type,
                )
            else:
                prompts, answers = sample_vllm(
                    llm,
                    tokenizer,
                    conversations,
                    temperature=question.temperature,
                    max_tokens=max_tokens,
                )

            df_data = [
                dict(question=q, prompt=p, answer=a, question_id=question.id)
                for q, a, p in zip(paraphrases, answers, prompts)
            ]
            df = pd.DataFrame(df_data)

            for score, judge in question.judges.items():
                scores = asyncio.run(
                    asyncio.gather(
                        *[
                            judge(question=q, answer=a)
                            for q, a in zip(paraphrases, answers)
                        ]
                    )
                )
                df[score] = scores

            outputs.append(df)
        outputs = pd.concat(outputs)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    print_results(outputs, trait)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
