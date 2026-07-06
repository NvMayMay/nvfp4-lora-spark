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

## Phase 0 — serve/merge spike (PREREQUISITE, zero training)

**Both reviewers: the merge/serve split is the single riskiest surface, and its outcome
sets the v1 default — so it runs FIRST.** The repo's runtime-LoRA evidence is text-model
only. Unknowns: does vLLM's `nano_nemotron_vl` path accept `--enable-lora` at all? Does its
LoRA machinery cover FP8LoRALinear on Mamba2 `in_proj`/`out_proj` (`families.py:239-241`)?
The Command-A precedent (base serves, `--enable-lora` crashes on a tied-embed arch) says
this is exactly where VLM serving breaks.

- Spike: existing nemotron vision-merged base + ANY existing text adapter → attempt vLLM
  `--enable-lora`. Zero training.
- Outcome drives the default:
  - runtime-LoRA works on the LLM half → default = **merge tower into base, keep LLM as
    runtime-LoRA** (the nybbloris thesis; preserves the swappable-LLM value prop).
  - it does not → default = **fully merge both halves** (document the regression vs the
    thesis) OR restrict `both`'s LLM targets to LoRA-servable modules.
- Deliverable: a decision + a serve recipe (or a documented limitation). Nothing in Phase 1
  is built until this lands.

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

## Phase 2 — save / merge / serve routing

- **Save:** structured record in `adapter_config.json` — text vs vision suffixes, both
  scopes, `include_projector`, base identity, and the LLM-half intent (`runtime`|`merge`).
  Not just a `train_target: both` tag (`train:1167` records only the string).
- **Splitter** (in `merge_vision_lora.py` or a new tool): emit a tower-only sub-adapter
  (→ merge into the bf16 tower → new merged VLM base) + an LLM-only sub-adapter; REWRITE the
  LLM sub-adapter's `base_model_name_or_path` to the merged-tower base + re-fingerprint
  (`train:696` writes the ORIGINAL base, wrong after tower-merge); its `target_modules`
  carries only text suffixes. Assert BOTH key-scopes present before splitting (R6). Must run
  BEFORE any rekey/merge tool — `merge_vision_lora.adapter_key_to_base_key` raises on
  text-backbone keys (`merge_vision_lora.py:118-123`) and `adapter_key_to_base_prefix` raises
  on tower keys (`families.py:606-611`); feeding a both-adapter to either is a hard crash
  today (loud, good).
- **Contract test:** round-trip a both-adapter through the splitter; `check_lora_binding`
  passes on both halves.

---

## Phase 3 — validation on Box A

- Extend `--dry-run` to synthesize an IMAGE batch for `both` (today it is text-only and skips
  the processor, `train:1209,1422-1430`), so the OOM preflight actually exercises tower
  activations + GPU image preproc (the R4 concern on nemotron's 0.55-util config). Or
  document the gap loudly.
- GPU: `nemotron_omni`, small mixed set (vqa-rad image rows + text-QA rows), r16, bs1, capped
  tiles/seq. Confirm: force-native; both wrap-halves > 0; grad-gate passes; loss drops on
  both halves; save → split → merge tower → serve → (runtime-LoRA or merged) LLM half →
  sanity inference. PLUMBING validation, not a metric claim. Periodic checkpoints per the
  training-safety policy.

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
