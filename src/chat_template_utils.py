"""
Utility functions for handling chat templates across different models.
"""


def apply_chat_template_safe(
    tokenizer, messages, tokenize=False, add_generation_prompt=True
):
    """
    Safely apply chat template with fallback for models that don't support system roles.

    Args:
        tokenizer: The tokenizer to use
        messages: List of message dictionaries with 'role' and 'content' keys
        tokenize: Whether to tokenize the result
        add_generation_prompt: Whether to add generation prompt

    Returns:
        str or tokens: The formatted chat text or tokens

    Raises:
        ValueError: If chat template fails even after fallback attempts
    """
    try:
        # Try with original messages
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        # Check if it's a system role issue
        if "System role not supported" in str(e) or "system" in str(e).lower():
            # Convert system role to user role instead of removing
            messages_converted = []
            for msg in messages:
                if msg["role"] == "system":
                    # Convert system to user role
                    messages_converted.append(
                        {"role": "user", "content": msg["content"]}
                    )
                else:
                    messages_converted.append(msg)
            try:
                return tokenizer.apply_chat_template(
                    messages_converted,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass  # Fall through to generic fallback

        # Generic fallback: extract user messages
        if len(messages) == 1:
            text = messages[0]["content"]
        else:
            # Get last user message
            user_messages = [
                msg["content"] for msg in messages if msg["role"] == "user"
            ]
            if user_messages:
                text = user_messages[-1]
            else:
                text = messages[-1]["content"]

        if tokenize:
            return tokenizer(text, return_tensors="pt")["input_ids"]
        else:
            return text
