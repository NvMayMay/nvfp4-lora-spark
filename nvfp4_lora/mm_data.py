"""Multimodal (image+text) chat data path for `--train-target vision` runs.

Text-only training keeps using `scripts/train_nvfp4_lora.py:ChatJsonlDataset`; this
module is exercised ONLY by a vision run, so a text run touches none of it (the
regression surface for text stays at zero). It is a library module (not trainer-inline)
so the no-model CPU test suite can exercise the row-shaping / masking / validation logic
with a STUB processor -- the only heavy dependency (the model's real `AutoProcessor`) is
injected, never constructed here.

Row contract (one JSON object per line), matching scoping doc section B:

    {"messages": [{"role": "user",
                   "content": [{"type": "image"},
                               {"type": "text", "text": "What is shown?"}]},
                  {"role": "assistant",
                   "content": [{"type": "text", "text": "A chest X-ray."}]}],
     "images": ["cases/cxr_00123.png"]}

`images` is an ordered sidecar list aligned 1:1 with the in-order `{"type":"image"}`
content parts; paths resolve relative to the JSONL file's directory (absolute paths are
used verbatim). The dataset opens them as RGB PIL images and hands them to the model's
own `AutoProcessor`, which -- not our code -- expands the single placeholder in the
rendered chat template into the full run of image tokens. Hand-expanding placeholders is
the classic silent-misalignment bug (scoping risk #3), so we never do it.

Loss is unchanged next-token cross-entropy on the assistant span, PLUS a belt-and-
suspenders pass that masks every image placeholder / control token to -100: the model
must never be trained to PREDICT an image token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Keys the processor emits that are NOT the padded text tensors: they are passed through
# to model(**batch) as-is (concatenated across the batch). `input_ids`/`attention_mask`
# are handled by the text-padding path; `labels` we compute ourselves.
_TEXT_KEYS = ("input_ids", "attention_mask", "labels")

# Best-effort image placeholder / control token strings across the reference VLMs
# (Pixtral: [IMG]/[IMG_BREAK]/[IMG_END]; Llama-4: <|image|>; InternVL/Nemotron:
# <image>/<img>/</img>). resolve_image_token_ids() maps whichever ones a tokenizer
# actually knows to ids; unknown tokens (mapping to unk / None) are dropped. Ids are
# resolved at runtime from the real processor, never hardcoded in a training row.
_KNOWN_IMAGE_CONTROL_TOKENS = (
    "[IMG]", "[IMG_BREAK]", "[IMG_END]",
    "<|image|>", "<|image_start|>", "<|image_end|>",
    "<image>", "<img>", "</img>",
)


# ---------------------------------------------------------------------------
# Pure, processor-free helpers (unit-testable with plain Python objects).
# ---------------------------------------------------------------------------

def count_image_parts(messages: Sequence[dict]) -> int:
    """Number of `{"type":"image"}` content parts across a message list.

    Only structured content (a list of parts) can carry images; a plain-string
    `content` contributes zero. Matches how the processor's chat template counts
    placeholders, so this is the count the `images` sidecar must equal.
    """
    n = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    n += 1
    return n


def resolve_image_paths(obj: dict, row_index: int, base_dir: Path) -> list[Path]:
    """Validate one row and return its resolved absolute image paths.

    Fail-fast with the ROW NUMBER (mirroring scripts/data_check.py's style) when
    the image-part count and `images` length disagree, or a file is missing -- a
    silently dropped / mis-aligned image corrupts the image-token fuse.
    """
    if "messages" not in obj:
        raise ValueError(f"row {row_index}: missing 'messages'")
    images = obj.get("images", []) or []
    n_parts = count_image_parts(obj["messages"])
    if n_parts != len(images):
        raise ValueError(
            f"row {row_index}: {n_parts} image content-part(s) but {len(images)} "
            f"path(s) in 'images' -- they must align 1:1 and in order"
        )
    resolved: list[Path] = []
    for rel in images:
        p = Path(rel)
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            raise ValueError(f"row {row_index}: image file not found: {p}")
        resolved.append(p)
    return resolved


def mask_labels(input_ids: Sequence[int], prompt_len: int,
                image_token_ids: Sequence[int]) -> list[int]:
    """Next-token-CE labels: -100 over the prompt AND every image token.

    `prompt_len` is the token count of everything up to (and excluding) the
    supervised assistant span; those positions are ignored. The image-token pass
    is belt-and-suspenders: even inside a supervised span, an image placeholder /
    control token is masked so the model is never trained to predict it.
    """
    img = set(int(i) for i in image_token_ids)
    labels = [int(t) for t in input_ids]
    for pos in range(len(labels)):
        if pos < prompt_len or labels[pos] in img:
            labels[pos] = -100
    return labels


def resolve_image_token_ids(processor: Any = None, config: Any = None,
                            extra: Sequence[int] = ()) -> set[int]:
    """Collect the image placeholder / control token ids from processor + config.

    Reads config fields (`image_token_index`, `image_token_id`, and the common
    vision start/end/break id fields when present) and maps the known image control
    token STRINGS through the tokenizer. Never hardcodes an id into a training row.
    """
    ids: set[int] = set(int(i) for i in extra)
    for attr in ("image_token_index", "image_token_id", "image_break_token_id",
                 "image_end_token_id", "vision_start_token_id", "vision_end_token_id"):
        val = getattr(config, attr, None) if config is not None else None
        if isinstance(val, int):
            ids.add(val)
    tok = getattr(processor, "tokenizer", processor)
    conv = getattr(tok, "convert_tokens_to_ids", None)
    unk = getattr(tok, "unk_token_id", None)
    if callable(conv):
        for name in _KNOWN_IMAGE_CONTROL_TOKENS:
            try:
                tid = conv(name)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.add(tid)
    return ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalJsonlDataset(Dataset):
    """Image+text chat JSONL -> {messages, images(PIL)} examples.

    Validation (row counts + file existence) runs eagerly at construction so a bad
    dataset fails before the GPU is touched; image DECODE is lazy (in __getitem__)
    to keep memory flat over a large set. The processor is NOT needed here -- the
    collate does the image-token expansion -- so this class is fully importable and
    testable without loading a model.
    """

    def __init__(self, path: str, *, max_examples: Optional[int] = None):
        self.path = Path(path)
        self.base_dir = self.path.parent
        self.rows: list[dict] = []
        self.image_paths: list[list[Path]] = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if max_examples is not None and len(self.rows) >= max_examples:
                    break
                obj = json.loads(line)
                paths = resolve_image_paths(obj, i, self.base_dir)
                self.rows.append(obj)
                self.image_paths.append(paths)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in self.image_paths[idx]]
        return {"messages": self.rows[idx]["messages"], "images": images}


# ---------------------------------------------------------------------------
# Processor-driven collate
# ---------------------------------------------------------------------------

class MultimodalCollator:
    """Turn {messages, images} examples into a model-ready batch via the processor.

    Injected with the model's own `AutoProcessor` (duck-typed: it must provide
    `apply_chat_template(messages, tokenize=False, add_generation_prompt=...)` and
    `__call__(text=..., images=..., return_tensors="pt")` returning at least
    `input_ids`/`attention_mask`, plus `pixel_values` (+ optional `image_sizes`)).
    The image-token expansion always comes from the processor -- never hand-rolled.

    Labels mask the prompt and every image token to -100 (see mask_labels). A
    per-example post-expansion length over `max_length` is a HARD ERROR (never a
    silent truncation that would corrupt a mid-image token run).
    """

    def __init__(self, processor: Any, *, image_token_ids: Sequence[int],
                 max_length: int, pad_token_id: int, label_pad: int = -100):
        self.processor = processor
        self.image_token_ids = set(int(i) for i in image_token_ids)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.label_pad = label_pad

    def _render(self, messages, *, add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _process(self, messages, images):
        text = self._render(messages, add_generation_prompt=False)
        kwargs = {"text": text, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs)

    def _render_len(self, messages, images, upto: int, *, add_generation_prompt: bool) -> int:
        """Token length of messages[:upto] rendered through the SAME processor.

        The prefix runs over exactly the images that appear within it, so image-token
        expansion is counted consistently with the full-example render. `start` of an
        assistant span uses add_generation_prompt=True (through the header); the `end`
        uses add_generation_prompt=False (through the assistant content).
        """
        prefix_msgs = messages[:upto]
        n_imgs = count_image_parts(prefix_msgs)
        text = self._render(prefix_msgs, add_generation_prompt=add_generation_prompt)
        kwargs = {"text": text, "return_tensors": "pt"}
        if n_imgs:
            kwargs["images"] = images[:n_imgs]
        enc = self.processor(**kwargs)
        return int(enc["input_ids"].shape[-1])

    def encode_example(self, example: dict) -> dict:
        """Encode ONE {messages, images} example into input_ids/labels/... tensors."""
        messages = example["messages"]
        images = example.get("images", []) or []
        enc = self._process(messages, images)
        input_ids = enc["input_ids"][0]
        seq_len = int(input_ids.shape[-1])
        if seq_len > self.max_length:
            raise ValueError(
                f"example expands to {seq_len} tokens > max_length={self.max_length}; "
                f"refusing to truncate (would cut a mid-image token run and corrupt the "
                f"fuse). Use a smaller image or a larger --max-length."
            )
        attn = enc["attention_mask"][0] if "attention_mask" in enc \
            else torch.ones_like(input_ids)

        # Supervise each assistant turn's span; mask everything else + image tokens.
        ids_list = input_ids.tolist()
        labels = [self.label_pad] * len(ids_list)
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            start = self._render_len(messages, images, i, add_generation_prompt=True)
            through = self._render_len(messages, images, i + 1, add_generation_prompt=False)
            end = min(through, len(ids_list))
            for pos in range(start, end):
                labels[pos] = ids_list[pos]
        # belt-and-suspenders: never predict an image token.
        for pos, tid in enumerate(ids_list):
            if tid in self.image_token_ids:
                labels[pos] = self.label_pad

        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # Pass every non-text processor key (pixel_values, image_sizes, aspect ratios,
        # ...) straight through VERBATIM -- the model's forward consumes them by name,
        # and the exact shape (leading example dim, per-image tiling) is the model's
        # own contract. Batching concatenates these along dim 0 (see _batch_extra); for
        # bs=1 (V0) the processor's output reaches the model untouched.
        for k, v in enc.items():
            if k in _TEXT_KEYS:
                continue
            out[k] = v
        return out

    def __call__(self, batch: list[dict]) -> dict:
        encoded = [self.encode_example(ex) for ex in batch]

        input_ids = pad_sequence([e["input_ids"] for e in encoded],
                                 batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([e["attention_mask"] for e in encoded],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([e["labels"] for e in encoded],
                              batch_first=True, padding_value=self.label_pad)
        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Non-text keys: concatenate the per-example image tensors along dim 0
        # (Pixtral/Llama-4 emit variable per-image patch/tile counts). bs=1 (V0)
        # passes the single example's tensors through unchanged; bs>1 stacking is
        # best-effort and re-validated with the real processors at V1.
        extra_keys = [k for k in encoded[0] if k not in _TEXT_KEYS]
        for k in extra_keys:
            vals = [e[k] for e in encoded if e.get(k) is not None]
            out[k] = _batch_extra(vals)
        return out


def _batch_extra(vals: list[Any]) -> Any:
    """Batch a processor's non-text values (pixel_values, image_sizes, ...)."""
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if all(isinstance(v, torch.Tensor) for v in vals):
        try:
            return torch.cat([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], dim=0)
        except RuntimeError:
            return list(vals)
    # Mixed / list-valued: flatten to one list of per-image tensors.
    flat: list[Any] = []
    for v in vals:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    return flat
