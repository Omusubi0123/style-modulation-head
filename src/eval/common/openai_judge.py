import asyncio
import math

from openai import AsyncOpenAI, RateLimitError

from src.settings import settings

openai = AsyncOpenAI(api_key=settings.openai_api_key)


class OpenAiJudge:
    """Evaluation judge class using OpenAI models

    OpenAI models tokenize 0-100 numbers as single tokens, so we can
    use logprobs to accurately get exactly one completion token.
    Other models may not behave this way, requiring different processing
    when used for evaluation.
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        """Initialize OpenAiJudge class

        Args:
            model (str): OpenAI model name to use
            prompt_template (str): Prompt template for evaluation
            eval_type (str): Evaluation type ("0_100", "0_10", "binary", "binary_text")
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
        """Calculate evaluation score based on question and answer

        Args:
            **kwargs: Keyword arguments for prompt template formatting

        Returns:
            float or None: Evaluation score (None for refusal)
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
        """Get log probabilities and return probability dictionary (simple logprobs request)

        Always samples 1 token and includes automatic retry for rate limit errors.

        Args:
            messages (list): Chat format message list

        Returns:
            dict: Token to probability mapping dictionary
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
        """Request full text completion

        Used for binary_text evaluation type and includes automatic retry
        for rate limit errors.

        Args:
            messages (list): Chat format message list

        Returns:
            str: Generated text response
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
        """Aggregate scores on 0-100 scale

        Does not explicitly check for refusals. Assumes it's not a refusal
        if total weight on numbers is at least 0.25.

        Args:
            score (dict): Token to probability mapping

        Returns:
            float or None: Aggregated score (None for refusal)
        """
        # NOTE: we don't check for refusals explicitly. Instead we assume that
        # if there's at least 0.25 total weight on numbers, it's not a refusal.
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
        """Aggregate scores on 0-10 scale

        Explicitly checks for refusals. Returns None if "REFUSAL" has highest probability.
        Otherwise aggregates scores from 0-9.

        Args:
            score (dict): Token to probability mapping

        Returns:
            float or None: Aggregated score (None for refusal)
        """
        # NOTE: we check for refusals explicitly.
        # If "REFUSAL" has the highest probability, we return None.
        # Otherwise, we aggregate scores from 0-9.
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
        """Aggregate binary scores

        Explicitly checks for refusals. Returns None if "REFUSAL" has higher
        probability than both YES and NO. Otherwise calculates score as
        P(YES) / (P(YES) + P(NO)). Returns None if combined probability of
        YES and NO is below threshold (0.25).

        Args:
            score (dict): Token to probability mapping

        Returns:
            float or None: Aggregated score (None for refusal)
        """
        # NOTE: we check for refusals explicitly.
        # If "REFUSAL" has a higher probability than both "YES" and "NO", we return None.
        # Otherwise, calculates a score P(YES) / (P(YES) + P(NO)).
        # Returns None if the combined probability of YES and NO is below a threshold (0.25).

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
        """Aggregate text-based binary scores

        Parses <answer> tags in response text to determine score.

        Args:
            response_text (str): Generated response text

        Returns:
            int or None: 0 (NO), 1 (YES), or None (refusal/invalid)
        """
        if "<answer>REFUSAL</answer>" in response_text:
            return None
        elif "<answer>NO</answer>" in response_text:
            return 0
        elif "<answer>YES</answer>" in response_text:
            return 1
        return None  # Invalid response

    async def __call__(self, **kwargs):
        """Method to call class instance as function

        Args:
            **kwargs: Keyword arguments to pass to judge() method

        Returns:
            float or None: Evaluation score
        """
        return await self.judge(**kwargs)
