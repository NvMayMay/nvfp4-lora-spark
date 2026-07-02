# CPU quick-win batch -- implementation plan (v2, post codex review)

Revised after a gpt-5.5 review against the actual source. Changes from v1: added a prerequisite
eval refactor (R0); all eval outputs keyed PER ADAPTER; fixed the A5 sign logic; rescoped C1/C2
and D3 out of the "quick" batch (they hide tokenizer/fixture work) into safe sub-deliverables.
Determinism policy (applies throughout): local `random.Random(0)` (never global seed); round only
at JSON output, aggregate on raw values; emit dicts with sorted keys; tests compare `json.dumps(
..., sort_keys=True)`. All changes additive / back-compat; existing headline numbers MUST NOT move
(reuse the same paired populations + row-mean NLL semantics as today).

## R0 (PREREQUISITE) -- factor a pure summary builder out of eval_retention.main()
- WHAT: extract `build_summary(per_example, models, *, no_nll, no_em, bootstrap_n=1000) -> summary`
  from main(), operating on the already-collected per-row results (raw, unrounded). main() keeps
  doing HTTP + row collection, then calls build_summary. A2/A3/A5/B3 all hang off this.
- WHY: codex blocking issue -- bolting CIs/per-db/divergence into main() duplicates logic and breaks
  determinism testability. Behavior-preserving refactor.
- ACCEPTANCE: existing eval output byte-identical (minus the new additive fields) on a fixed
  per_example fixture; `from eval_retention import build_summary` works without a server.
- FILES: scripts/eval_retention.py.

## A2 -- bootstrap CIs on the EM/NLL delta (PER ADAPTER)
- ADDRESSES: headline deltas ship with no uncertainty.
- FIX: in build_summary, for each adapter in models[1:], bootstrap-resample the PAIRED population
  (rows where base AND that adapter both produced the metric -- same set the point delta uses;
  NLL uses row-mean, NOT token-weighted), 1000 resamples via a local `random.Random(0)`, 95% CI by
  explicit percentile (lower=sorted[int(0.025*n)], upper=sorted[int(0.975*n)] with clamped indices).
  Emit `summary["em_delta_ci_vs_base"]={model:[lo,hi]|None}` and `["nll_delta_ci_vs_base"]` likewise.
  Empty paired set -> None; single row -> degenerate [d,d]. Round only on output.
- FILES: scripts/eval_retention.py.  TEST: fixed per_example -> CI deterministic + brackets point delta.

## A3 -- per-slice (per-db) breakdown (PER ADAPTER)
- ADDRESSES: one easy DB could carry the whole delta (Spider: 140 train vs 20 eval dbs).
- FIX: (a) prep_spider.py writes top-level `db_id` per row (NOT in messages -> model never sees it;
  ChatJsonlDataset reads only obj["messages"], so training unaffected). (b) eval main() carries
  `rec["db_id"]=row.get("db_id")` only when present. (c) build_summary, when db_id present, emits
  `summary["per_db"]={db:{ "n_em":, "n_nll":, per-adapter {em_base,em_ft,em_delta,nll_base,nll_ft,
  nll_delta} }}` following the SAME paired rules as global. Sorted db keys. Absent db_id -> omit section.
- FILES: scripts/prep_spider.py, scripts/eval_retention.py.  TEST: with/without db_id; paired counts correct.

## A5 -- metric-divergence detector (CORRECTED SIGNS)
- ADDRESSES: NLL improves while EM regresses (or vice-versa) -- silent quality regression.
- FIX: per adapter, with nll_delta = adapter-minus-base (negative=better) and em_delta (positive=better):
  divergence iff `(nll_delta < -0.01 and em_delta < -0.02)` OR `(nll_delta > 0.01 and em_delta > 0.02)`.
  SUPPRESS when EM is unreliable for that adapter (all/majority empty generations -- reuse the existing
  empty-gen detection) so this never fires on a measurement failure. Skip when either delta is None.
  Append a loud WARNING.
- FILES: scripts/eval_retention.py.  TEST: divergent -> warn; concordant -> none; all-empty -> suppressed.

## B3 -- eval scoring/aggregation determinism test
- ADDRESSES: "deterministic eval" claim unproven in CI; pairs with the (GPU) parity gate later.
- FIX: unit test calls build_summary (R0) twice on identical cached per_example fixtures and asserts
  `json.dumps(s, sort_keys=True)` byte-identical. Exercises A2 bootstrap (local RNG), A3, A5. No HTTP.
- FILES: tests/test_eval_summary_determinism.py (new).  TEST: itself.

## E2 -- run-metadata bundle
- ADDRESSES: a run dir is a black box later; auditability.
- FIX: `build_run_meta(args, coverage) -> dict` then write run_meta.json AFTER output_dir exists and
  coverage is known (~train_nvfp4_lora.py:558/572-588). Contents: arg/config snapshot; git SHA via
  guarded subprocess (None if no git/binary); sha256 of train+val files (tolerate missing val + dry-run
  with no train-file); versions via importlib.metadata.version() for torch/transformers/peft (NOT import);
  coverage summary. Sorted keys. On resume: write resume_meta.json instead of overwriting; warn if args
  differ from the original.
- FILES: scripts/train_nvfp4_lora.py.  TEST: build_run_meta on stub args -> expected keys; helpers no-op
  cleanly when git/files absent (no GPU, no model).

## E3 -- richer metrics.jsonl
- ADDRESSES: can't tell if a 20h run is healthy; need running ETA.
- FIX: `build_metrics_row(step, total_updates, window_supervised_tokens, wall_elapsed, recent_upd_s,
  loss_window_mean)` adding: supervised-tokens/s (accumulate `(labels != -100).sum()` over the accum
  window, NOT input tokens), updates/s, window-mean loss (current log is only the last micro-batch),
  cuda allocated/reserved + free (guarded by torch.cuda.is_available() AND try/except), host-mem-available
  (psutil if importable else None), running ETA over SUCCESSFUL update_steps only (skip nonfinite windows
  ~796-818). Pure helper so it's unit-testable without the loop.
- FILES: scripts/train_nvfp4_lora.py.  TEST: build_metrics_row on stub counters -> fields present; cuda/
  psutil paths return None off-GPU.

## C0 (RESCOPED from C1/C2) -- extract a pure chat-encode/report helper ONLY
- WHY codex: a full `data-check` CLI is NOT a CPU quick win -- importing train_nvfp4_lora has heavy
  import-time side effects (set_alloc_conf at import), tokenizer trust_remote_code / Mistral paths, and
  ChatJsonlDataset eagerly materializes + silently drops rows. So the QUICK deliverable is just the
  foundation:
- FIX: extract `encode_chat_example(messages, tokenizer, max_length) -> {n_tokens, n_supervised,
  dropped_reason|None, truncated:bool}` into a LIGHT module (no torch/accelerate import-time side effects;
  pure tokenization + masking math mirroring ChatJsonlDataset). Refactor ChatJsonlDataset to use it (so
  there's one masking source of truth). The full `nybbloris data-check` CLI is DEFERRED to a follow-up
  ticket that builds on this helper (tokenizer-load + render handled there).
- FILES: new nvfp4_lora/chat_encode.py (or similar light module); scripts/train_nvfp4_lora.py uses it.
- TEST: helper on a tiny messages list + a stub tokenizer -> correct counts / drop reason. (No model.)

## D3 (RESCOPED) -- document the PEFT round-trip behavior, don't fake a fixture
- WHY codex: peft.PeftModel.from_pretrained needs a REAL compatible base; a toy fixture proves nothing
  about the actual native/NVFP4 adapter. Not a CPU quick win as a real test.
- FIX: (a) a small unit test asserting `_save_adapter_atomic` key SHAPE (base_model.model.{name}.lora_
  {A,B}.weight + target_modules suffixes), which IS CPU-cheap and guards the save format; (b) a
  `@pytest.mark.integration` + `xfail(strict=False)` placeholder that documents the intended stock-PEFT
  round-trip as a known-untested path, to be filled when a real minimal base fixture exists. No README
  edit yet -- that follows the real verdict.
- FILES: tests/test_adapter_save_shape.py (new).  TEST: itself.

## Sequencing for implementation
R0 first (unblocks A2/A3/A5/B3) -> A2, A3, A5 -> B3 (tests them) -> E2, E3 (independent) -> C0 -> D3.
Run `pytest tests/` (currently 141) green after; ~7-9 new tests. Drop nothing silently -- C1/C2 full
CLI and the real D3 round-trip are explicitly DEFERRED follow-ups, recorded in credibility_backlog.md.
