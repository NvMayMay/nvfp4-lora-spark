from __future__ import annotations

from nvfp4_lora.chat_encode import encode_chat_example


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _Enc([ord(c) for c in text])


def test_encode_chat_example_reports_counts_and_truncation():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]

    encoded = encode_chat_example(messages, StubTokenizer(), max_length=30)

    assert encoded["n_tokens"] == 30
    assert encoded["n_supervised"] > 0
    assert encoded["dropped_reason"] is None
    assert encoded["truncated"] is True
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == len(encoded["attention_mask"])


def test_encode_chat_example_reports_drop_reason_when_no_supervised_tokens():
    messages = [
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "ok"},
    ]

    encoded = encode_chat_example(messages, StubTokenizer(), max_length=10)

    assert encoded["n_supervised"] == 0
    assert encoded["dropped_reason"] == "no_supervised_tokens"
