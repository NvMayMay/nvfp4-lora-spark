# `--train-target both`: joint LLM + vision-tower LoRA (FINAL plan)

One training run that LoRA-adapts BOTH the vision tower/projector (perception) AND the
LLM backbone (reasoning/output format), from a mixed dataset (image+text rows interleaved
with text-only rows). Use case: document-QC ("read the signatures" = tower; "emit the
structured verdict" = LLM) — unreachable by `text` or `vision` alone.

This is the post-review plan. It supersedes the scope draft. Two external reviews (codex
gpt-5.5 @high, fable) verified the draft against the code; both confirmed the core thesis
and surfaced correctness bugs + one strategic reorder, all folded in below.

---

## Verdict from review

**Core thesis is structurally sound:** load the text backbone as in `text` mode, load +
materialize the vision tower as in `vision` mode, attach LoRA to both, freeze base weights,
and let autograd route gradient by graph membership (no per-sample freeze toggling). The
existing machinery genuinely composes — `freeze_all_then_enable_lora` already enables every
`(NVFP4|FP8|BF16)LoRALinear` A/B (`train:530-535`), the collator tolerates image-free rows
(`mm_data.py:259`), and `replace_bf16_targets` uses leaf-eq + scope-anchor matching
(`loader.py:782-788`).

**The risk is NOT autograd — it is (a) scope/mode routing on the training side and (b)
whether the serving stack can runtime-LoRA a VLM's LLM half at all.** The draft
under-specified both. Nine correctness fixes + one reorder follow.

---

## Corrections to the draft (things it got wrong or under-specified)

**C1. Registry `freeze` is NOT empty for vision families — verified.** mistral3
`("vision_tower","multi_modal_projector")` (`families.py:91`), llama4
`("vision_model","multi_modal_projector")` (`families.py:210`), nemotron_omni
`("vision_model","mlp1","sound_encoder","sound_projection")` (`families.py:277`). KEEPING
registry freeze for `both` is correct (it keeps nemotron's SOUND tower frozen), but it means
the load_model freeze loop (`train:473-480`) runs AFTER `replace_bf16_targets` and freezes
the freshly created tower `lora_A`/`lora_B`. So **`freeze_all_then_enable_lora` is
LOAD-BEARING for `both`, not a safety belt** — it re-enables them. Document it so nobody
"optimizes away" the call and silently ships a frozen tower.

**C2. `both` MUST force `lora_mode="native"` — the #1 correctness bug (both reviewers).**
`detect_lora_mode` returns `"peft"` when the targets carry no NVFP4/FP8 (`loader.py:262`).
Nemotron's attention q/k/v are BF16 (`families.py:240-241`), so a `both` run targeting them
classifies as `peft`, and then: `replace_bf16_targets` never runs (gated on native,
`train:436`) so NEITHER half is wrapped, and `attach_peft_lora` wraps a PeftModel whose
`base_model.model.` re-pathing breaks the native save/grad-gate machinery. Force native
before load, exactly as vision does (`train:1120-1125`).

**C3. bs>1 mixed batch is order-dependent DATA LOSS, not "best-effort."** `extra_keys` is
derived from `encoded[0]` only (`mm_data.py:323`); a batch whose FIRST row is text-only
silently drops `pixel_values` for the whole batch while `input_ids` still hold image tokens
→ silent corruption / opaque forward crash. **Hard-error `batch_size>1` for `both`** (one
`if` in the collator or CLI validation). Homogeneous bucketing is deferred (it interacts
with the seeded-shuffle resume replay, `train:1249-1253,1528-1537`).

**C4. Grad-gate strictness is ASYMMETRIC (fable, more correct than the draft's "≥1 both").**
- VISION half: keep ALL-nonzero `lora_B`. That is the exact guarantee the gate exists for
  (a partially severed graph / mis-scoped wrap). The tower is dense; ALL-nonzero is valid.
- TEXT half: must be ≥1, NEVER all. Nemotron stores routed experts as per-expert `nn.Linear`
  (`train:390-396`), so text suffixes (`up_proj`/`down_proj`) attach LoRA to every expert and
  only the routed subset gets grad per batch — an ALL-nonzero text check hard-fails every
  healthy MoE run.
- Fire on the first batch with `"pixel_values" in batch`, not blindly the first backward.
- Fix the `".lora_B" in n` substring bug (`train:576`): it also matches expert
  `lora_B_gate_up`/`lora_B_down`. Use suffix-anchored matching, and split scopes via the
  view's `_vision_peft_scope`/`_projector_scopes` regexes, NOT hardcoded name prefixes
  (mistral3 tower keys are `model.vision_tower.*`, nemotron's are `vision_model.*`).

**C5. Expert-LoRA silent zero-delta save — verified latent bug (fable).**
`freeze_all_then_enable_lora` only re-enables the three LoRALinear classes (`train:531`); it
does NOT know `NVFP4Experts3D.lora_A_gate_up/lora_B_down` (`train:1349`). But
`_save_adapter_atomic` saves expert LoRA by `lora_r>0` with NO `requires_grad` check
(`train:667`). So a `both`/`vision` run on a `moe_experts_class` vision family (llama4,
`families.py:210,218`) with `--expert-lora-r>0` creates expert LoRA, freezes it, excludes it
from the optimizer (`train:1411`), and STILL saves a zero-delta expert adapter with an
`expert_lora` config block. **v1 hard-rejects `--expert-lora-r` with `both`; separately fix
the same latent bug in `vision` mode.**

**C6. Coverage-inventory pollution (fable).** `build_target_inventory` (`loader.py:153`)
drops modules by the view's `skip_st_prefixes`. The `both` view un-skips the vision prefixes
(the tower must load), so a text-suffix inventory now counts TOWER modules (Pixtral's tower
has `q_proj/.../down_proj` leaves, `families.py:104-106`), inflating `tot_bf16` and
corrupting `target_coverage.json` — the artifact the repo's QC story leans on. `both` needs
TWO restricted inventories: text suffixes with the ORIGINAL skip list; vision suffixes
restricted to `vision_st_prefixes`; merged under distinct keys.

**C7. Mid-run crash on long text rows (fable).** Text mode DROPS over-length rows at dataset
construction (`train:262-269`); the mm path RAISES lazily at collate (`mm_data.py:263-268`).
In a mixed corpus one over-length text row kills a multi-hour run at a random step (the
exact class the checkpoint-safety policy exists for). Eagerly length-validate/drop text-only
rows at `MultimodalJsonlDataset` construction, or permit truncation for image-free rows
(safe: no image-token run to corrupt).

**C8. De-dup between paired passes is automatic (verified).** All three LoRALinear classes
subclass `nn.Module`, not `nn.Linear` (`linear.py:526`), and `replace_bf16_targets` only
collects `isinstance(mod, nn.Linear)` (`loader.py:790`), so pass B inherently skips anything
pass A wrapped. No explicit `BF16LoRALinear` guard needed — but assert/​document the
invariant. Pass A MUST pass `projector_scopes=()` (else projector Linears get wrapped under
the text pass and mis-attributed).

---

## Phase 0 — serve/merge spike (RESOLVED 2026-07-06, zero training)

**Question:** can vLLM 0.22.1 runtime-LoRA the LLM half of the nemotron VLM (so a `both`
adapter could serve as merged-tower base + swappable LLM LoRA)?

**Answer: NO on vLLM 0.22.1** — settled by source introspection alone (no serve/GPU needed):
- The merged base's arch resolves to the multimodal wrapper class: `registry.py:500`
  `"NemotronH_Nano_Omni_Reasoning_V3": ("nano_nemotron_vl", "NemotronH_Nano_VL_V2")`.
- `NemotronH_Nano_VL_V2` does NOT declare `SupportsLoRA` (`nano_nemotron_vl.py:902` bases =
  `HasInnerState, IsHybrid, SupportsMultiModal, SupportsMultiModalPruning`); vLLM's own
  `supports_lora(cls)` returns `False`.
- vLLM hard-errors at startup: `lora_model_runner_mixin.py:37-38`
  `if not supports_lora(model): raise ValueError(f"{cls} does not support LoRA yet.")`. So
  `vllm serve <merged> --enable-lora` fails before generating (the Command-A failure shape).

**Key nuance:** the capability exists at the LLM level — standalone `NemotronHForCausalLM`
returns `supports_lora=True`. It is the multimodal WRAPPER that doesn't expose LoRA, not a
Mamba/hybrid limitation. A wrapper subclass declaring `SupportsLoRA` + delegating the inner
LLM's `packed_modules_mapping`/`embedding_modules` could unlock it later — the same shape as
the Command-A tied-embed `__getattr__` delegation patch (cohere memory). That is a Phase-3+
vLLM-patch item, not v1.

**v1 default decision (revises §3.6): FULLY MERGE BOTH HALVES.** The splitter emits a
tower-only sub-adapter merged into the bf16 tower AND an LLM-only sub-adapter merged into the
LLM weights; serve the merged VLM as a plain model (no `--enable-lora`).
- Quality: merging the LLM half is LOSSLESS iff its targets are bf16 (nemotron's `q/k/v` are
  bf16) — bf16 + bf16 delta stays bf16, so the repo's "merge-into-4bit erases the fine-tune"
  concern does NOT apply to a bf16-attention `both` run. It DOES apply if the LLM targets are
  FP8 (`o_proj`, Mamba `in/out_proj`) or NVFP4 (routed experts); for those the wrapper-patch
  runtime-LoRA path is the quality-preserving option, deferred to Phase 3+.
- Practical guidance for `both` on nemotron: target the bf16 attention (`q_proj,k_proj,
  v_proj`) so the merge is clean; document that FP8/NVFP4 LLM targets take a merge quality hit
  until the wrapper patch lands.

---

## Phase 1 — training-side core (after Phase 0)

1. **`family_view("both")` branch** (`families.py`): reuse the vision load-inversions
   (un-skip `vision_st_prefixes`, un-meta tower mem-prefixes, `+ vision_st_to_model`, stash
   `_projector_scopes`); KEEP registry `freeze` (C1); carry `_text_peft_scope`,
   `_vision_peft_scope`, `_projector_scopes`; pin `view["peft_scope"] = fam["peft_scope"]`
   (text) so stray consumers behave text-like; `_train_target="both"`; require
   `family_supports_vision`.
2. **Mode + inventory:** force `lora_mode="native"` for `both` before load (C2); both-branch
   `build_target_inventory` with two restricted inventories (C6).
3. **`load_model`:** generalize `is_vision` → tower-load for `target in {vision, both}`;
   `native_targets = text suffixes` for `both`; TWO paired `replace_bf16_targets` passes —
   pass A `(text_suffixes, _text_peft_scope, projector_scopes=())`, pass B
   `(vision_suffixes, _vision_peft_scope, projector_scopes=_projector_scopes)` (C8);
   per-half wrap asserts (each > 0); call `freeze_all_then_enable_lora` (load-bearing, C1);
   un-gate the compat hooks to `{vision, both}` — `mm_embed_grad_hook` is REQUIRED in `both`
   too (embeddings stay frozen, the in-place tower-feature scatter still hits the
   frozen-leaf-view trap).
4. **CLI:** add `"both"`; resolve TWO suffix lists (text = `--target-modules`; vision =
   `--vision-target-modules` or family default); hard-reject `--expert-lora-r` (C5);
   hard-error `batch_size>1` (C3).
5. **Data path:** `both` → mm collator/dataset; eager length-validate/drop text-only rows at
   construction (C7); collator hard-errors a batch that mixes image + image-free rows;
   dataset-build assert ≥1 image row exists.
6. **Grad-gate:** asymmetric strictness + first-image-batch arming + suffix-anchored `lora_B`
   matching + view-regex scope split (C4).
7. **Un-gate every remaining `== "vision"` site to `{vision, both}`** (full list from
   review): `train:340,403` (`is_vision` load), `1095` (log), `1106-1112` (suffix
   resolution), `1120-1125` (force-native), `1209` (processor/collator), `1238-1241`
   (dataset), `1288` (compat hooks), `1327-1334` (freeze/enable + `n_vis` assert), `1465`
   (gate arming), `loader.py:153` (inventory).

---

## Phase 2 — save / merge / serve routing (DONE 2026-07-06)

- **Save (Phase 1):** the native adapter_config carries a `both` block — text vs vision
  suffixes, both scopes, `include_projector`, base identity — so the splitter routes keys
  without re-deriving scope membership (`_save_adapter_atomic`, `both_meta`).
- **Splitter — `scripts/split_both_adapter.py` (new):** reads the unified both-adapter and
  its `both` block, classifies each `base_model.model.<path>.lora_{A,B}.weight` key by module
  path against the vision scope + projector scopes, and writes `<out>/tower/` +
  `<out>/llm/`, each a standard PEFT-shaped sub-adapter (r + lora_alpha + its half's
  target_modules) that its own merge tool accepts. Self-contained (scopes come from the
  config, no model/family load) so it is CPU-unit-tested. **R6 guard:** refuses if EITHER
  scope is empty (a half-trained / mis-scoped adapter) — loud SystemExit, never a silent
  half-merge. It prints the fully-merge command sequence (tower first, then LLM against the
  tower-merged base — the Phase-0 v1 default; no `--enable-lora`).
  - Note: the two existing merge tools take `--base-model-dir` on the CLI, so the LLM base
    pointer is advisory; the merge ORDER (documented in the tool's output) is what makes the
    LLM half merge against the tower-merged base.
- **Contract test — `tests/test_split_both_adapter.py` (5 cases):** module-path parsing,
  scope partitioning (disjoint + complete), a full round-trip (tower/llm keys land in the
  right sub-adapter with valid configs), the non-both refusal, and the empty-scope R6 guards.
  Full suite 372 passed.
- **Deferred to Phase 3 (needs a real base + GPU):** actually running the two merges on a
  trained both-adapter and confirming each sub-adapter binds/merges cleanly against the
  nemotron base (the "binding" validation; there is no standalone `check_lora_binding`
  function in-repo — it is a merge-run + serve-parity check).

---

## Phase 3 — validation on Box A (DONE 2026-07-06, end-to-end green)

Full pipeline validated on `nemotron_omni`, mixed set (40 vqa-rad image rows + 20 text-QA
rows), r16, bs1, 1-tile, 30 updates: **train → split → merge tower → merge LLM → serve →
image + text inference.** PLUMBING validation, not a metric claim.

- **Train:** force-native fired (bf16 q/k/v were peft → native); paired passes wrapped 18
  text-bf16 + 130 tower/projector; `both_freeze_enable` per-half assert (text 36 / vision
  260); **grad-gate passed** (18/18 text lora_B + 130 vision lora_B); ran all 30 updates
  through MIXED image+text batches. Saved a 25 MB both-adapter (260 tower + 36 LLM keys) with
  the `both` config block.
- **Two runtime findings + fixes (what CPU tests could not catch):**
  1. nemotron's training `forward()` MANDATES `pixel_values` and runs the tower
     unconditionally — a text-only row crashed run 1. Fix: `mm_text_only_bypass` (family flag)
     + `build_text_only_bypass_forward` — image batch hits the real forward; text-only batch
     runs `language_model` directly (tower out of the graph, no wasted tower compute).
  2. `merge_lora_into_nvfp4` cannot merge the bf16 LLM half of a VLM: it is single-backbone
     (no backbone prefix in a multi-backbone VLM) AND pre-rejects bf16 targets. Fix: the LLM
     half is bf16, so it merges through the SAME dequant-free `merge_vision_lora` path via a
     new `--prefix-pair language_model.:language_model.` override. `split_both_adapter` now
     emits that command.
- **Split → merge:** splitter → 260 tower / 36 LLM sub-adapters; tower merge (130 bf16
  targets) → tower-merged base; LLM merge (18 bf16 q/k/v, `--prefix-pair`) → fully-merged
  VLM. delta/base ratios 0.002–0.018 (sane light-FT deltas).
- **Serve:** the fully-merged VLM serves as a PLAIN model (no `--enable-lora`); arch resolved,
  ModelOpt FP8+NVFP4 quant preserved (merge only touched bf16 weights). Sanity: a vqa-rad
  brain scan → "Brain"; "capital of France?" → "Paris". Both paths work.
- **Not done (documented gap):** the `--dry-run` OOM preflight still synthesizes a text-only
  batch and skips the processor (`train:1209,1422-1430`), so it does not exercise tower
  activations / GPU image preproc — a real both run at higher seq/tiling could still OOM
  where dry-run said OK. Left as a follow-up.

---

## Deferred (OUT of v1, documented as such)

- bs>1 / homogeneous bucketing (interacts with seeded-shuffle resume replay).
- per-half LoRA rank/alpha (paired passes keep the door open).
- expert-LoRA + `both` (hard-rejected in v1; additive later).
- audio/video joint (image only).

## Confirmed non-issues (verified by review — no action)

Tied-embedding re-tie (`train:484-485`) touches nothing the passes wrap; GC sub-module
fallback already enables `language_model` + `vision_model` (`train:1379-1401`); dist init is
family-gated not target-gated (`train:1278`); optimizer collection (`train:1411`) correctly
runs after freeze/enable.

## Effort

Phase 0 is the real gating unknown (a serve capability nothing in this repo has exercised).
Phases 1-2 are medium — mostly composition of already-"both"-friendly machinery plus the
routing/inventory fixes above. Phase 3 is GPU-headroom gated.
