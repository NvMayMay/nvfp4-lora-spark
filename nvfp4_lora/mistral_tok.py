"""mistral_common (tekken) tokenizer wrapper for training.

The HF LlamaTokenizerFast conversion shipped with some Mistral NVFP4 repacks
(e.g. Mistral-Small-3.2-24B-Instruct-2506-NVFP4) is broken in current
transformers: it produces different token ids than the model's native tekken
tokenizer and cannot round-trip (drops spaces / leaks the byte-BPE marker).
Training through it teaches the model wrong token boundaries.

This wrapper uses the model's own tekken.json via mistral_common so the trainer
tokenizes exactly as the model was pretrained. It exposes only what the trainer
needs: pad_token_id, vocab_size, and encode_chat() which returns assistant-loss-
masked input_ids/labels (mirroring ChatJsonlDataset._encode but token-native).

Detection: use this whenever a checkpoint dir contains tekken.json.
"""
from __future__ import annotations

from pathlib import Path

import torch


def has_tekken(model_dir) -> bool:
    return (Path(model_dir) / "tekken.json").is_file()


class MistralCommonTokenizer:
    is_mistral_common = True

    def __init__(self, model_dir):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self._src = Path(model_dir)
        self._mt = MistralTokenizer.from_file(str(Path(model_dir) / "tekken.json"))
        self._raw = self._mt.instruct_tokenizer.tokenizer
        self.eos_id = int(self._raw.eos_id)
        # No dedicated pad token; pad with eos and mask via attention_mask + labels=-100.
        self.pad_token_id = self.eos_id
        self.vocab_size = int(self._raw.n_words)

    # ---- chat encoding (mistral_common request API) ----
    def _to_messages(self, messages):
        from mistral_common.protocol.instruct.messages import (
            AssistantMessage,
            SystemMessage,
            UserMessage,
        )

        ctor = {"system": SystemMessage, "user": UserMessage, "assistant": AssistantMessage}
        last = len(messages) - 1
        out = []
        for i, m in enumerate(messages):
            cls = ctor[m["role"]]
            if m["role"] == "assistant" and i == last:
                out.append(cls(content=m["content"], prefix=True))  # no trailing EOS yet
            else:
                out.append(cls(content=m["content"]))
        return out

    def _encode_request(self, messages):
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        return self._mt.encode_chat_completion(
            ChatCompletionRequest(messages=self._to_messages(messages))
        ).tokens

    def encode_chat(self, messages, max_length):
        """Return assistant-loss-masked example or None if no supervised tokens.

        Supervises the final assistant turn (the ICH corpus is single-turn).
        prompt = messages up to the last assistant; full = prompt + assistant
        (prefix=True, no EOS); we append EOS so the model learns to stop.
        """
        ai = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        if ai < 0:
            return None
        prompt_tokens = self._encode_request(messages[:ai])
        full_tokens = self._encode_request(messages)  # last assistant is prefix=True
        if full_tokens[: len(prompt_tokens)] != prompt_tokens:
            # tekken should make the prompt a strict prefix; bail loudly if not.
            raise RuntimeError(
                "mistral_common prompt is not a prefix of the full sequence; "
                "assistant-span masking would be wrong."
            )
        assistant_tokens = full_tokens[len(prompt_tokens):]
        input_ids = full_tokens + [self.eos_id]
        labels = [-100] * len(prompt_tokens) + assistant_tokens + [self.eos_id]
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        if all(l == -100 for l in labels):
            return None  # assistant fell entirely beyond max_length
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, ids):
        return self._raw.decode(list(ids))

    def save_pretrained(self, dest):
        """Copy the native tekken tokenizer files next to the saved adapter so the
        adapter dir is self-contained (serve with vLLM --tokenizer-mode mistral)."""
        import shutil

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for fn in ("tekken.json", "tokenizer_config.json",
                   "special_tokens_map.json", "chat_template.jinja"):
            src = self._src / fn
            if src.is_file():
                shutil.copy2(src, dest / fn)
