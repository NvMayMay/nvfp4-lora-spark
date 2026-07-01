"""Lightweight chat-example tokenization and assistant-label masking helpers."""
from __future__ import annotations


def _tokenize(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _render(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def encode_chat_example(messages, tokenizer, max_length: int) -> dict:
    """Tokenize one chat example and mark assistant tokens as supervised."""
    full_text = _render(tokenizer, messages, add_generation_prompt=False)
    input_ids = _tokenize(tokenizer, full_text)
    truncated = len(input_ids) > max_length
    if not input_ids:
        return {
            "n_tokens": 0,
            "n_supervised": 0,
            "dropped_reason": "empty_tokenization",
            "truncated": False,
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }

    labels = [-100] * len(input_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_ids = _tokenize(tokenizer, _render(tokenizer, messages[:index], add_generation_prompt=True))
        through_ids = _tokenize(tokenizer, _render(tokenizer, messages[: index + 1], add_generation_prompt=False))
        start = len(prefix_ids)
        end = min(len(through_ids), len(input_ids))
        for pos in range(start, end):
            labels[pos] = input_ids[pos]

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    n_supervised = sum(1 for label in labels if label != -100)
    dropped_reason = "no_supervised_tokens" if n_supervised == 0 else None
    return {
        "n_tokens": len(input_ids),
        "n_supervised": n_supervised,
        "dropped_reason": dropped_reason,
        "truncated": truncated,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }
