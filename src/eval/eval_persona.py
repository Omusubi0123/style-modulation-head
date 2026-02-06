"""
eval_persona.py - Persona evaluation with optional steering

Supports multiple steering modes:
- No steering (coef=0): Uses vLLM for fast inference
- Layer output steering: Add to transformer block output (residual stream)
- Block-level steering: Add to specific block submodule (attn/mlp input/output)
- Head-level steering: Add to specific attention heads' O projection input

Layer index convention:
  --layer k corresponds to model.layers[k] (0-indexed transformer block).
  vector_path のテンソルも同じ規約: tensor[k] = model.layers[k] に対応するベクトル.

Usage examples:
    # No steering (data extraction)
    uv run python src/eval/eval_persona.py \\
        --model Qwen/Qwen2.5-7B-Instruct --trait evil \\
        --output_path results/evil_extract.csv --version extract

    # Residual stream (layer output) steering
    uv run python src/eval/eval_persona.py \\
        --model Qwen/Qwen2.5-7B-Instruct --trait evil \\
        --output_path results/evil_steer.csv \\
        --vector_path vectors/evil_response_avg_diff.pt \\
        --layer 19 --coef 2.0

    # Block-level steering (e.g. attention output)
    uv run python src/eval/eval_persona.py \\
        --model Qwen/Qwen2.5-7B-Instruct --trait evil \\
        --output_path results/evil_block.csv \\
        --vector_path vectors/evil_response_avg_diff_attn_output.pt \\
        --layer 19 --coef 2.0 --steering_target attn_output

    # Head-level steering
    uv run python src/eval/eval_persona.py \\
        --model Qwen/Qwen2.5-7B-Instruct --trait evil \\
        --output_path results/evil_head.csv \\
        --vector_path vectors/evil_response_avg_diff_attn_pre_o_proj.pt \\
        --layer 19 --coef 7.0 --steering_target head --head_indices "2,4,27"
"""

import asyncio
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.eval.common.judge import run_judge_evaluations
from src.eval.common.loaders import load_persona_questions
from src.eval.common.question import Question
from src.eval.common.utils import print_results
from src.eval.model_utils import load_model, load_vllm_model
from src.eval.sampling import (
    sample_vllm,
    sample_with_block_steering,
    sample_with_head_steering,
    sample_with_steering,
)

# Valid block-level steering targets
BLOCK_STEERING_TARGETS = {
    "attn",
    "mlp",
    "attn_layernorm",
    "mlp_layernorm",
    "attn_output",
    "mlp_output",
}

ALL_STEERING_TARGETS = {"layer_output", "head"} | BLOCK_STEERING_TARGETS


def _generate_responses(
    llm,
    tokenizer,
    conversations: list[list[dict]],
    coef: float,
    vector: torch.Tensor | None,
    layer: int | None,
    steering_target: str,
    head_indices: list[int] | None,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    steering_type: str,
):
    """Generate model responses with the appropriate steering method.

    Args:
        llm: Language model
        tokenizer: Tokenizer
        conversations: List of conversations
        coef: Steering coefficient
        vector: Steering vector (1-D)
        layer: Layer index (0-based transformer block)
        steering_target: Steering target type
        head_indices: Head indices (for steering_target="head")
        batch_size: Batch size
        temperature: Sampling temperature
        max_tokens: Maximum generation tokens
        steering_type: When to steer ("response", "prompt", "all")

    Returns:
        tuple: (prompts, answers)
    """
    needs_steering = coef != 0 and vector is not None and layer is not None
    if steering_target == "head":
        needs_steering = needs_steering and head_indices is not None

    if not needs_steering:
        return sample_vllm(
            llm,
            tokenizer,
            conversations,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if steering_target == "layer_output":
        return sample_with_steering(
            llm,
            tokenizer,
            conversations,
            vector,
            layer,
            coef,
            bs=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    elif steering_target == "head":
        return sample_with_head_steering(
            llm,
            tokenizer,
            conversations,
            vector,
            layer,
            head_indices,
            coef,
            bs=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    elif steering_target in BLOCK_STEERING_TARGETS:
        return sample_with_block_steering(
            llm,
            tokenizer,
            conversations,
            vector,
            layer,
            coef,
            block_steering_type=steering_target,
            bs=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    else:
        raise ValueError(
            f"Invalid steering_target: {steering_target}. "
            f"Must be one of {sorted(ALL_STEERING_TARGETS)}"
        )


async def eval_batched(
    questions: list[Question],
    llm: object,
    tokenizer: object,
    coef: float,
    vector: torch.Tensor | None = None,
    layer: int | None = None,
    steering_target: str = "layer_output",
    head_indices: list[int] | None = None,
    batch_size: int = 100,
    n_per_question: int = 5,
    max_concurrent_judges: int = 4,
    max_tokens: int = 1000,
    steering_type: str = "response",
):
    """Process all questions in batch for fast inference.

    Args:
        questions: List of Question objects
        llm: Language model
        tokenizer: Tokenizer
        coef: Steering coefficient
        vector: Steering vector
        layer: Layer index (0-based transformer block)
        steering_target: Steering target type
        head_indices: Head indices (for head steering)
        batch_size: Batch size
        n_per_question: Samples per question
        max_concurrent_judges: Max concurrent judge evaluations
        max_tokens: Max generation tokens
        steering_type: When to steer ("response", "prompt", "all")

    Returns:
        list[pd.DataFrame]: Evaluation results per question
    """
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

    # Generate answers using the appropriate steering method
    prompts, answers = _generate_responses(
        llm,
        tokenizer,
        all_conversations,
        coef,
        vector,
        layer,
        steering_target,
        head_indices,
        batch_size,
        questions[0].temperature,
        max_tokens,
        steering_type,
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
    model: str,
    trait: str,
    output_path: str,
    coef: float = 0,
    vector_path: str | None = None,
    layer: int | None = None,
    steering_target: str = "layer_output",
    head_indices: list[int] | int | tuple | str | None = None,
    steering_type: str = "response",
    max_tokens: int = 1000,
    n_per_question: int = 5,
    batch_process: bool = True,
    max_concurrent_judges: int = 4,
    batch_size: int = 100,
    persona_instruction_type: str | None = None,
    assistant_name: str | None = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    version: str = "extract",
):
    """Execute persona evaluation with optional steering.

    Args:
        model: Model name to evaluate
        trait: Trait name to evaluate
        output_path: CSV file path to save results
        coef: Steering coefficient (0 = no steering)
        vector_path: Path to steering vector file (.pt)
        layer: Layer index (0-based transformer block). tensor[layer] is loaded.
        steering_target: Steering target type:
            "layer_output"   - Transformer block output (residual stream)
            "attn"           - Attention block input (post-LayerNorm)
            "mlp"            - MLP block input (post-LayerNorm)
            "attn_layernorm" - LayerNorm input before attention
            "mlp_layernorm"  - LayerNorm input before MLP
            "attn_output"    - Attention block output
            "mlp_output"     - MLP block output
            "head"           - Attention head O projection input (requires head_indices)
        head_indices: Head indices for head steering (comma-separated string,
                      int, or list). Only used when steering_target="head".
        steering_type: When to apply steering ("response", "prompt", "all")
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
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}, skipping...")
        df = pd.read_csv(output_path)
        print_results(df, trait)
        return

    # Parse head_indices from various input formats
    if isinstance(head_indices, str):
        head_indices = [int(x.strip()) for x in head_indices.split(",")]
    elif isinstance(head_indices, int):
        head_indices = [head_indices]
    elif isinstance(head_indices, tuple):
        head_indices = list(head_indices)

    # Validate steering_target
    if steering_target not in ALL_STEERING_TARGETS:
        raise ValueError(
            f"Invalid steering_target: {steering_target}. "
            f"Must be one of {sorted(ALL_STEERING_TARGETS)}"
        )

    print(f"Output path: {output_path}")
    print(f"Steering target: {steering_target}")
    if head_indices is not None:
        print(f"Head indices: {head_indices}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temperature = 0.0 if n_per_question == 1 else 1.0

    # Determine if steering is needed
    needs_steering = coef != 0 and vector_path is not None and layer is not None
    if steering_target == "head":
        needs_steering = needs_steering and head_indices is not None

    # Load model and optionally load steering vector
    if needs_steering:
        llm, tokenizer = load_model(model)
        vector = torch.load(vector_path, weights_only=False)[layer]
        print(f"Loaded vector from {vector_path}[{layer}], shape: {vector.shape}")
    else:
        llm, tokenizer = load_vllm_model(model)
        vector = None

    # Load questions
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
                steering_target,
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
            paraphrases, conversations = question.get_input(n_per_question)

            prompts, answers = _generate_responses(
                llm,
                tokenizer,
                conversations,
                coef,
                vector,
                layer,
                steering_target,
                head_indices,
                batch_size,
                question.temperature,
                max_tokens,
                steering_type,
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

    # Add head metadata if applicable
    if head_indices is not None and steering_target == "head":
        outputs["head_indices"] = str(head_indices)

    outputs.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print_results(outputs, trait)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
