import asyncio
import math

from openai import AsyncOpenAI, RateLimitError

from src.config import setup_credentials

# Set up credentials and environment
config = setup_credentials()
openai = AsyncOpenAI()


class OpenAiJudge:
    """OpenAIモデルを使用した評価判定クラス

    OpenAIモデルは0-100の数値を単一トークンとしてトークン化するため、
    logprobsを使って正確に1つの完了トークンを取得できる。
    他のモデルは必ずしもこのような動作をしないため、判定に使用する際は
    異なる処理が必要。
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        """OpenAiJudgeクラスの初期化

        Args:
            model (str): 使用するOpenAIモデル名
            prompt_template (str): 評価用のプロンプトテンプレート
            eval_type (str): 評価タイプ（"0_100", "0_10", "binary", "binary_text"）
        """
        self.model = model
        assert eval_type in [
            "0_100",
            "0_10",
            "binary",
            "binary_text",
        ], "eval_type must be either 0_100 or binary"
        self.eval_type = eval_type

        if self.eval_type == "0_100":
            self.aggregate_score = self._aggregate_0_100_score
        elif self.eval_type == "0_10":
            self.aggregate_score = self._aggregate_0_10_score
        elif self.eval_type == "binary":
            self.aggregate_score = self._aggregate_binary_score
        elif self.eval_type == "binary_text":
            self.aggregate_score = self._aggregate_binary_text_score
        else:
            raise ValueError(f"Invalid eval_type: {self.eval_type}")

        self.prompt_template = prompt_template

    async def judge(self, **kwargs):
        """質問と回答に基づいて評価スコアを算出する

        Args:
            **kwargs: プロンプトテンプレートのフォーマット用キーワード引数

        Returns:
            float or None: 評価スコア（拒否の場合はNone）
        """
        messages = [dict(role="user", content=self.prompt_template.format(**kwargs))]
        if self.eval_type == "binary_text":
            response_text = await self.query_full_text(messages)
            score = self.aggregate_score(
                response_text
            )  # aggregate_score is _aggregate_binary_text_score
        else:
            logprobs = await self.logprob_probs(messages)
            score = self.aggregate_score(
                logprobs
            )  # aggregate_score is one of the other three
        return score

    async def logprob_probs(self, messages) -> dict:
        """ログ確率を取得して確率辞書を返す（単純なlogprobsリクエスト）

        常に1トークンをサンプリングし、レート制限エラーに対する
        自動リトライ機能を含む。

        Args:
            messages (list): チャット形式のメッセージリスト

        Returns:
            dict: トークンと確率のマッピング辞書
        """
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                completion = await openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                    seed=0,
                )
                try:
                    choice = completion.choices[0]
                    lp = getattr(choice, "logprobs", None)
                    # Handle cases where logprobs or its content/top_logprobs are missing/empty
                    if (
                        not lp
                        or not getattr(lp, "content", None)
                        or len(lp.content) == 0
                    ):
                        return {}
                    first_content = lp.content[0]
                    if not getattr(first_content, "top_logprobs", None):
                        return {}
                    logprobs = first_content.top_logprobs
                except (IndexError, AttributeError, KeyError, TypeError):
                    # This can happen if the model doesn't return logprobs for this request
                    return {}

                result = {}
                for el in logprobs:
                    result[el.token] = float(math.exp(el.logprob))

                return result

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e

                # Extract wait time from error message if available
                wait_time = base_delay * (2**attempt)  # Exponential backoff
                if "Please try again in" in str(e):
                    try:
                        # Try to extract the exact wait time from the error message
                        import re

                        match = re.search(r"Please try again in (\d+)ms", str(e))
                        if match:
                            wait_time = max(wait_time, int(match.group(1)) / 1000.0)
                    except:
                        pass

                print(
                    f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}"
                )
                await asyncio.sleep(wait_time)

        return {}

    async def query_full_text(self, messages) -> str:
        """完全なテキスト補完をリクエストする

        binary_text評価タイプで使用され、レート制限エラーに対する
        自動リトライ機能を含む。

        Args:
            messages (list): チャット形式のメッセージリスト

        Returns:
            str: 生成されたテキスト応答
        """
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                completion = await openai.chat.completions.create(
                    model=self.model, messages=messages, temperature=0, seed=0
                )
                try:
                    return completion.choices[0].message.content
                except (IndexError, AttributeError):
                    return ""

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e

                # Extract wait time from error message if available
                wait_time = base_delay * (2**attempt)  # Exponential backoff
                if "Please try again in" in str(e):
                    try:
                        # Try to extract the exact wait time from the error message
                        import re

                        match = re.search(r"Please try again in (\d+)ms", str(e))
                        if match:
                            wait_time = max(wait_time, int(match.group(1)) / 1000.0)
                    except:
                        pass

                print(
                    f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}"
                )
                await asyncio.sleep(wait_time)

        return ""

    def _aggregate_0_100_score(self, score: dict) -> float:
        """0-100スケールのスコアを集約する

        拒否を明示的にチェックせず、数値の総重みが0.25以上であれば
        拒否ではないと仮定する。

        Args:
            score (dict): トークンと確率のマッピング

        Returns:
            float or None: 集約されたスコア（拒否の場合はNone）
        """
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_0_10_score(self, score: dict) -> float:
        """0-10スケールのスコアを集約する

        拒否を明示的にチェックし、"REFUSAL"が最高確率の場合はNoneを返す。
        そうでなければ0-9のスコアを集約する。

        Args:
            score (dict): トークンと確率のマッピング

        Returns:
            float or None: 集約されたスコア（拒否の場合はNone）
        """
        #   NOTE: we check for refusals explcitly.
        #   If "REFUSAL" has the highest probability, we return None.
        #   Otherwise, we aggregate scores from 0-9.
        if "REFUSAL" in score and score["REFUSAL"] > max(
            score.get(str(i), 0) for i in range(10)
        ):
            return None

        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 9:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:  # Similar threshold as 0_100
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_binary_score(self, score: dict) -> float:
        """二値スコアを集約する

        拒否を明示的にチェックし、"REFUSAL"がYESとNOの両方より高い確率の場合はNoneを返す。
        そうでなければP(YES) / (P(YES) + P(NO))でスコアを計算する。
        YESとNOの合計確率が閾値（0.25）を下回る場合はNoneを返す。

        Args:
            score (dict): トークンと確率のマッピング

        Returns:
            float or None: 集約されたスコア（拒否の場合はNone）
        """
        #   NOTE: we check for refusals explicitly.
        #   If "REFUSAL" has a higher probability than both "YES" and "NO", we return None.
        #   Otherwise, calculates a score P(YES) / (P(YES) + P(NO)).
        #   Returns None if the combined probability of YES and NO is below a threshold (0.25).

        yes_prob = score.get("YES", 0.0)
        no_prob = score.get("NO", 0.0)
        refusal_prob = score.get("REFUSAL", 0.0)

        # If REFUSAL has a higher probability than both YES and NO, consider it a refusal.
        if refusal_prob > yes_prob and refusal_prob > no_prob:
            return None

        denominator = yes_prob + no_prob

        # If the combined probability of YES and NO is too low (e.g., model outputted something else,
        # or was not confident in YES/NO), return None.
        if (
            denominator < 0.25
        ):  # Using 0.25 to be consistent with other aggregation methods
            return None

        return yes_prob / denominator

    def _aggregate_binary_text_score(self, response_text: str) -> bool:
        """テキストベースの二値スコアを集約する

        応答テキスト内の<answer>タグを解析してスコアを決定する。

        Args:
            response_text (str): 生成された応答テキスト

        Returns:
            int or None: 0（NO）、1（YES）、またはNone（拒否/無効）
        """
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None  # Invalid response

    async def __call__(self, **kwargs):
        """クラスインスタンスを関数として呼び出すためのメソッド

        Args:
            **kwargs: judge()メソッドに渡すキーワード引数

        Returns:
            float or None: 評価スコア
        """
        return await self.judge(**kwargs)
