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
from src.visualize.projection_tracker import ProjectionTrackerBlock
from src.visualize.projection_visualizer import ProjectionVisualizer

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
    coef,
    bs=100,
    top_p=1,
    max_tokens=1000,
    temperature=1,
    min_tokens=1,
    steering_type="response",
    block_steering_type="attn_input",
    track_projections=False,
    renorm_after_steering: bool = False,
):
    """ステアリングベクトルを使用してモデルの応答を生成する（block入力用）

    Args:
        model: 使用する言語モデル
        tokenizer: トークナイザー
        conversations (list): 会話データのリスト
        vector: ステアリングベクトル
        layer (int): 適用するレイヤー番号（0-based index、0=1層目）
        coef (float): ステアリング係数
        bs (int): バッチサイズ（デフォルト: 100）
        top_p (float): nucleus samplingパラメータ（デフォルト: 1）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        temperature (float): サンプリング温度（デフォルト: 1）
        min_tokens (int): 最小生成トークン数（デフォルト: 1）
        steering_type (str): ステアリングタイプ（デフォルト: "response"）
        block_steering_type (str): block steeringタイプ（"attn", "mlp", "attn_layernorm", "mlp_layernorm", "attn_output", "mlp_output"）
                - "attn": attention input
                - "mlp": MLP input
                - "attn_layernorm": attention layernorm input
                - "mlp_layernorm": MLP layernorm input
                - "attn_output": attention output
                - "mlp_output": MLP output
        track_projections (bool): 投影値を追跡するか（デフォルト: False）
        renorm_after_steering (bool): ステアリング後にノルムをリノームするか（デフォルト: False）

    Returns:
        tuple: (プロンプトリスト, 応答リスト, 投影値リスト[track_projections=Trueの場合])
    """
    tracker = ProjectionTrackerBlock(
        model, tokenizer, vector, layer, steering_type, block_steering_type
    )

    prompts, outputs, projections = tracker.generate_with_projections(
        conversations,
        coef=coef,
        bs=bs,
        top_p=top_p,
        max_tokens=max_tokens,
        temperature=temperature,
        min_tokens=min_tokens,
        renorm_to_original_norm=renorm_after_steering,
    )

    if track_projections:
        return prompts, outputs, projections
    else:
        return prompts, outputs


def load_jsonl(path):
    """JSONLファイルを読み込んで辞書のリストとして返す

    Args:
        path (str): 読み込むJSONLファイルのパス

    Returns:
        list: 各行がパースされたJSONオブジェクトのリスト
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


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
        judge: str = "gpt-4o",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        """質問クラスの初期化

        Args:
            id (str): 質問の一意識別子
            paraphrases (list[str]): 質問の言い換えリスト
            judge_prompts (dict): 判定用プロンプトの辞書
            temperature (float): サンプリング温度（デフォルト: 1）
            system (str): システムメッセージ（オプション）
            judge (str): 判定用モデル名（デフォルト: "gpt-4o"）
            judge_eval_type (str): 判定タイプ（デフォルト: "0_100"）
            **ignored_extra_args: その他の無視される引数
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

    def get_input(self, n_per_question):
        """質問から入力データを生成する

        Args:
            n_per_question (int): 質問あたりのサンプル数

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

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=1000,
        n_per_question=100,
        steering_type="last",
        block_steering_type="attn_input",
        lora_path=None,
        track_projections=False,
        renorm_after_steering: bool = False,
    ):
        """質問に対するモデルの応答を生成し、評価する

        Args:
            llm: 使用する言語モデル
            tokenizer: トークナイザー
            coef (float): ステアリング係数
            vector: ステアリングベクトル（オプション）
            layer (int): ステアリングレイヤー（0-based index、0=1層目、オプション）
            max_tokens (int): 最大生成トークン数（デフォルト: 1000）
            n_per_question (int): 質問あたりのサンプル数（デフォルト: 100）
            steering_type (str): ステアリングタイプ（デフォルト: "last"）
            lora_path (str): LoRAアダプターのパス（オプション）
            track_projections (bool): 投影値を追跡するか（デフォルト: False）

        Returns:
            pandas.DataFrame: 評価結果を含むデータフレーム
        """
        paraphrases, conversations = self.get_input(n_per_question)
        projections_data = None

        if vector is not None and layer is not None and track_projections:
            # ステアリングあり/なしに関わらず統一されたパスを使用
            result = sample_steering(
                llm,
                tokenizer,
                conversations,
                vector,
                layer,
                coef,  # 0でも非0でも対応
                temperature=self.temperature,
                max_tokens=max_tokens,
                steering_type=steering_type,
                block_steering_type=block_steering_type,
                track_projections=track_projections,
                renorm_after_steering=renorm_after_steering,
            )
            if track_projections:
                prompts, answers, projections_data = result
            else:
                prompts, answers = result
        else:
            # 従来のステアリングのみ（投影なし）
            result = sample_steering(
                llm,
                tokenizer,
                conversations,
                vector,
                layer,
                coef,
                temperature=self.temperature,
                max_tokens=max_tokens,
                steering_type=steering_type,
                block_steering_type=block_steering_type,
                track_projections=False,
                renorm_after_steering=renorm_after_steering,
            )
            prompts, answers = result

        df_data = [
            dict(question=question, prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ]

        # 投影値データを追加
        if projections_data is not None:
            for i, projections in enumerate(projections_data):
                df_data[i]["projections"] = projections

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
    """英語の不定冠詞を決定するヘルパー関数

    Args:
        word (str): 対象の単語

    Returns:
        str: "a"または"an"
    """
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
    """ペルソナ評価用の質問リストを読み込み、Questionオブジェクトのリストを作成する

    Args:
        trait (str): 評価する特性名
        temperature (float): サンプリング温度（デフォルト: 1）
        persona_instructions_type (str): ペルソナ指示タイプ（"pos", "neg", またはNone）
        assistant_name (str): アシスタント名（オプション）
        judge_model (str): 判定用モデル名（デフォルト: "gpt-4.1-mini-2025-04-14"）
        eval_type (str): 評価タイプ（デフォルト: "0_100"）
        version (str): データバージョン（デフォルト: "eval"）

    Returns:
        list: Questionオブジェクトのリスト
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
    batch_size=100,
    n_per_question=100,
    max_concurrent_judges=4,
    max_tokens=1000,
    steering_type="last",
    block_steering_type="attn_input",
    lora_path=None,
    track_projections=False,
    renorm_after_steering: bool = False,
):
    """すべての質問を一括処理して高速な推論を実現する

    Args:
        questions (list): Questionオブジェクトのリスト
        llm: 使用する言語モデル
        tokenizer: トークナイザー
        coef (float): ステアリング係数
        vector: ステアリングベクトル（オプション）
        layer (int): ステアリングレイヤー（オプション）
        batch_size (int): バッチサイズ（デフォルト: 100）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 100）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 4）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        steering_type (str): ステアリングタイプ（デフォルト: "last"）
        lora_path (str): LoRAアダプターのパス（オプション）
        track_projections (bool): 投影値を追跡するか（デフォルト: False）

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
    all_projections_data = None

    result = sample_steering(
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
        block_steering_type=block_steering_type,
        track_projections=track_projections,
        renorm_after_steering=renorm_after_steering,
    )
    if track_projections:
        prompts, answers, all_projections_data = result
    else:
        prompts, answers = result

    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task

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

        # Add projections data if available
        if all_projections_data is not None:
            q_projections = [all_projections_data[j] for j in indices]
            for idx, projections in enumerate(q_projections):
                df_data[idx]["projections"] = projections

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
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        """セマフォアで制御された判定実行関数

        Args:
            task_idx (int): タスクのインデックス
            judge: 判定者オブジェクト
            question_text (str): 質問テキスト
            answer (str): 回答テキスト

        Returns:
            tuple: (タスクインデックス, 判定結果)
        """
        async with semaphore:
            # Add small delay between requests to avoid rate limiting
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
            all_results[task_idx] = result  # Store result in correct position
            pbar.update(1)

    # Distribute results back to the appropriate dataframes
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
    steering_type="response",
    block_steering_type="attn_input",
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
    track_projections=False,
    visualization_dir=None,
    renorm_after_steering: bool = False,
):
    """評価YAMLファイルからすべての質問に対してモデルを評価する

    Args:
        model (str): 評価するモデル名
        trait (str): 評価する特性名
        output_path (str): 結果を保存するCSVファイルパス
        coef (float): ステアリング係数（デフォルト: 0）
        vector_path (str): ステアリングベクトルのパス（オプション）
        layer (int): ステアリングレイヤー（オプション）
        steering_type (str): ステアリングタイプ（デフォルト: "response"）
        max_tokens (int): 最大生成トークン数（デフォルト: 1000）
        n_per_question (int): 質問あたりのサンプル数（デフォルト: 10）
        batch_process (bool): バッチ処理を使用するか（デフォルト: True）
        max_concurrent_judges (int): 最大同時判定数（デフォルト: 100）
        persona_instruction_type (str): ペルソナ指示タイプ（オプション）
        assistant_name (str): アシスタント名（オプション）
        judge_model (str): 判定用モデル名（デフォルト: "gpt-4.1-mini-2025-04-14"）
        version (str): データバージョン（デフォルト: "extract"）
        overwrite (bool): 既存ファイルを上書きするか（デフォルト: False）
        track_projections (bool): 投影値を追跡するか（デフォルト: False）
        visualization_dir (str): 可視化ファイルの出力ディレクトリ（オプション）

    Returns:
        None
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for trait in [trait, "coherence"]:
            threshold = 50
            print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0
    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        # ベクトルファイルのindex layerを取得（0=1層目、1=2層目、...）
        vector = torch.load(vector_path, weights_only=False)[layer]
    elif track_projections and vector_path is not None and layer is not None:
        # coef=0でも投影追跡が有効な場合はベクトルを読み込む
        llm, tokenizer = load_model(model)
        lora_path = None
        # ベクトルファイルのindex layerを取得（0=1層目、1=2層目、...）
        vector = torch.load(vector_path, weights_only=False)[layer]
    else:
        llm, tokenizer, lora_path = load_vllm_model(model)
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
                block_steering_type=block_steering_type,
                lora_path=lora_path,
                track_projections=track_projections,
                renorm_after_steering=renorm_after_steering,
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
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                        block_steering_type=block_steering_type,
                        lora_path=lora_path,
                        track_projections=track_projections,
                        renorm_after_steering=renorm_after_steering,
                    )
                )
            )
        outputs = pd.concat(outputs)
    # 投影値の可視化を作成（追跡が有効でベクトルが存在する場合）
    if (
        track_projections
        and vector_path is not None
        and "projections" in outputs.columns
    ):
        if visualization_dir is None:
            visualization_dir = os.path.join(
                os.path.dirname(output_path), "projections"
            )

        # 非空の投影値を持つ行を取得
        valid_projections = []
        for proj in outputs["projections"]:
            if isinstance(proj, list) and len(proj) > 0:
                valid_projections.append(proj)

        if valid_projections:
            print(
                f"Creating projection visualizations for {len(valid_projections)} responses..."
            )
            visualizer = ProjectionVisualizer()
            title_suffix = "with steering" if coef != 0 else "no steering"
            saved_files = visualizer.create_projection_graphs(
                valid_projections,
                visualization_dir,
                trait,
                layer,
                coef,
                steering_type,
                title_suffix,
            )
            visualizer.print_graph_summary(saved_files)

    outputs.to_csv(output_path, index=False)
    print(output_path)
    for trait in [trait, "coherence"]:
        print(f"{trait}:  {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
