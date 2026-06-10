"""ChatJsonlDataset + collate_batch: assistant-only masking, truncation, NaN guard.

A deterministic char-level stub tokenizer stands in for a real HF tokenizer so the
test needs no model files. apply_chat_template wraps each message as
"<role>content</role>" and (when add_generation_prompt) appends a bare "<assistant>"
open tag; __call__ maps each character to its ord(). This makes byte offsets exact,
so we can assert which positions are supervised.
"""
from __future__ import annotations

import json

import torch


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class StubTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "dataset always renders text then tokenizes separately"
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _Enc([ord(c) for c in text])


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_assistant_only_label_masking(train_mod, tmp_path):
    rows = [{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=10_000)
    assert len(ds) == 1
    item = ds[0]
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()

    # The prefix that must be masked is "<user>hi</user>" + generation prompt "<assistant>".
    prefix = "<user>hi</user><assistant>"
    n_prefix = len(prefix)
    # Everything before the assistant content is -100.
    assert all(l == -100 for l in labels[:n_prefix])
    # Every supervised label equals the corresponding input id (loss only on assistant).
    for i, l in enumerate(labels):
        assert l == -100 or l == input_ids[i]
    # There is at least one supervised token, and supervision starts exactly at the
    # assistant content boundary.
    supervised = [i for i, l in enumerate(labels) if l != -100]
    assert supervised, "expected supervised assistant tokens"
    assert supervised[0] == n_prefix
    assert item["attention_mask"].tolist() == [1] * len(input_ids)


def test_multi_turn_masks_only_assistant_turns(train_mod, tmp_path):
    rows = [{"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]}]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=10_000)
    item = ds[0]
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    # Reconstruct which characters are supervised by decoding ord() back to chars.
    supervised_text = "".join(chr(input_ids[i]) for i, l in enumerate(labels) if l != -100)
    # Both assistant payloads (and their closing tags) are supervised; user/system are not.
    assert "A1" in supervised_text and "A2" in supervised_text
    assert "U1" not in supervised_text and "U2" not in supervised_text
    assert "S" not in supervised_text


def test_truncation_to_max_length(train_mod, tmp_path):
    rows = [{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}]
    path = _write_jsonl(tmp_path, rows)
    # Full sequence is 40 tokens; assistant content starts at offset 26. max_length=30
    # keeps a handful of supervised tokens while dropping the tail.
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=30)
    item = ds[0]
    assert len(item["input_ids"]) == 30
    assert len(item["labels"]) == 30
    assert len(item["attention_mask"]) == 30
    # Still has supervision (so it is not dropped) but fewer tokens than the full turn.
    assert any(l != -100 for l in item["labels"].tolist())


def test_nan_guard_drops_example_with_no_supervised_tokens(train_mod, tmp_path):
    # Long user turn pushes the assistant content entirely past max_length, so after
    # truncation every label is -100. The dataset must drop the example (return None)
    # to avoid a NaN loss that would poison the adapter.
    rows = [{"messages": [
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "ok"},
    ]}]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=10)
    assert len(ds) == 0


def test_empty_tokenization_returns_no_items(train_mod, tmp_path):
    # An empty render yields no input_ids -> the example is dropped.
    class EmptyTokenizer(StubTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return ""

    rows = [{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, EmptyTokenizer(), max_length=128)
    assert len(ds) == 0


def test_max_examples_limit(train_mod, tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"}]}
        for i in range(5)
    ]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=10_000, max_examples=3)
    assert len(ds) == 3


def test_collate_batch_pads_and_masks(train_mod, tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "a"},
                      {"role": "assistant", "content": "bb"}]},
        {"messages": [{"role": "user", "content": "cccc"},
                      {"role": "assistant", "content": "d"}]},
    ]
    path = _write_jsonl(tmp_path, rows)
    ds = train_mod.ChatJsonlDataset(path, StubTokenizer(), max_length=10_000)
    batch = train_mod.collate_batch([ds[0], ds[1]], pad_token_id=7)

    b, t = batch["input_ids"].shape
    assert b == 2
    # Both rows padded to the longest sequence.
    len0, len1 = len(ds[0]["input_ids"]), len(ds[1]["input_ids"])
    assert t == max(len0, len1)
    # Padding conventions: input_ids -> pad id, labels -> -100, attention_mask -> 0.
    shorter = 0 if len0 < len1 else 1
    assert batch["input_ids"][shorter, -1].item() == 7
    assert batch["labels"][shorter, -1].item() == -100
    assert batch["attention_mask"][shorter, -1].item() == 0
    assert batch["input_ids"].dtype == torch.long
