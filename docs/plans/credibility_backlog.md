# Credibility infrastructure backlog (Tier 1/1.5 of the comprehensiveness roadmap)

Ticket-level decomposition of the roadmap's credibility layer. Each item:
WHAT / DONE (testable acceptance) / WHERE (files) / EFFORT (S<1d, M~1-3d, L>3d) / GPU? / DEPENDS.
This is the remaining-work backlog after the v1.5 credibility refresh.

## Shipped in v1.5 (not active backlog)
- **A2 bootstrap CIs on the delta.** SHIPPED: `scripts/eval_retention.py` emits paired
  `em_delta_ci_vs_base` and `nll_delta_ci_vs_base` from `build_summary`; covered by summary tests.
- **A3 per-slice breakdown.** SHIPPED: `prep_spider.py` preserves `db_id`, and `eval_retention.py`
  emits sorted `summary["per_db"]` with paired counts and EM/NLL deltas when `db_id` is present.
- **A5 metric-divergence detector.** SHIPPED: eval summaries warn on favorable NLL with regressed EM,
  suppressing the warning for all-empty generations.
- **B0 runtime apply-check.** SHIPPED: `scripts/serve_apply_check.py` proves runtime adapter application
  via prompt-echo logprob deltas, catching loaded-but-no-op adapters.
- **B0 binding-contract hardening.** SHIPPED: the serve contract is wrapped-model, format, and
  backend-aware, including the Qwen3.5 wrapped-path no-op cases in tests.
- **E2 run-metadata bundle.** SHIPPED: `train_nvfp4_lora.py` writes `run_meta.json` or `resume_meta.json`
  with config, git SHA, dataset hashes, package versions, and coverage.
- **E3 baseline metrics stream.** PARTIAL: `metrics.jsonl` now carries supervised tokens/s, updates/s,
  ETA, CUDA alloc/reserved/free, and host memory. Per-phase timing is still open below.

## Epic A -- Eval honesty & leakage (defends every headline number)
- **A1 contamination check.** WHAT: script reporting exact-question + n-gram (8-gram) overlap and
  exact-db/schema overlap between train and eval JSONL; WARN over threshold. DONE: run on Spider
  train vs dev, emit overlap report JSON + loud WARN; documented number in REPRODUCE. WHERE:
  scripts/check_contamination.py (new). EFFORT: M. GPU: no. DEPENDS: -.
- **A4 held-out-db generalization split.** WHAT: verify/doc that eval dbs are unseen in train; add a
  "novel-db" subset metric. DONE: documented + a novel-db EM number. WHERE: prep_spider.py, REPRODUCE.
  EFFORT: S. GPU: no. DEPENDS: shipped per-db/db_id support.
- **A6 publish CI/slice numbers in docs.** WHAT: update public reproduction docs and committed result
  summaries to show the v1.5 CI/per-db outputs. DONE: README/REPRODUCE headline deltas include CIs and
  point to per-db slices. WHERE: README.md, REPRODUCE_SPIDER.md, results/spider/. EFFORT: S. GPU: no.
  DEPENDS: shipped A2/A3.

## Epic B -- Train<->serve parity + determinism (trust backbone; gate for everything after)
- **B1 numerical parity harness.** WHAT: tiny synthetic adapter; assert train-side LoRA delta ==
  serve-side applied delta within tol, per target type INCL experts (catches rekey/merge drift). The
  v1.5 apply-check proves the adapter changes the runtime forward pass; this ticket upgrades that into
  a per-target numerical equality gate. DONE: a GPU test that fails on a deliberately corrupted rekey.
  WHERE: tests/ + scripts/parity_*.py (extend serve_parity_vllm.py and/or serve_apply_check.py).
  EFFORT: L. GPU: yes (full serve). DEPENDS: shipped apply-check.
- **B2 same-seed determinism (train).** WHAT: two 5-update micro-runs, same seed -> identical adapter
  weights / identical loss trajectory. DONE: CI test. WHERE: tests/. EFFORT: M. GPU: small (or CPU mock).
  DEPENDS: -.
- **B3 same-eval determinism.** WHAT: eval_retention twice -> identical numbers (temp 0 already). DONE:
  live eval assert + doc note. WHERE: tests/ or a doc check. EFFORT: S. GPU: no (against a live server).
  DEPENDS: pure `build_summary` determinism is already covered.

## Epic C -- data-check / data doctor (catch "trained on the wrong thing")
- **C1 `nybbloris data-check`.** WHAT: given chat JSONL + model dir: render chat template on N samples
  (show masked vs unmasked spans), token-length histogram, count rows DROPPED by truncation at max-len,
  flag empty-assistant-span rows, report assistant-token coverage %. DONE: CLI subcommand emits report;
  WARN if truncation-drop > threshold. WHERE: nybbloris/cli.py + reuse ChatJsonlDataset logic. EFFORT: M.
  GPU: no. DEPENDS: -.
- **C2 hashes in the report.** WHAT: tokenizer/template/model hashes emitted (feeds the manifest). DONE:
  report includes hashes. WHERE: with C1. EFFORT: S. GPU: no. DEPENDS: C1.

## Epic D -- Adapter provenance + portability
- **D1 adapter manifest.** WHAT: stamp manifest.json into every adapter + merged dir at save: base
  repo+rev+safetensors hashes, tokenizer hash, chat-template hash, quant layout version, family-registry
  version, target coverage, rank/alpha/dropout, train git SHA, dep versions, masking/packing, eval
  summary, merge-compat. DONE: every saved dir has manifest.json; inspector reads it. WHERE:
  train_nvfp4_lora.py save path + nybbloris/cli.py inspect. EFFORT: M. GPU: no. DEPENDS: C2 (hashes).
- **D2 format versioning + compat reject.** WHAT: schema-version field + serve-time check rejecting an
  incompatible adapter with a NAMED reason. DONE: serve rejects a mismatched adapter clearly. WHERE:
  manifest + serve/inspect. EFFORT: M. GPU: small. DEPENDS: D1.
- **D3 PEFT round-trip (README:344 time bomb).** WHAT: actually test peft.PeftModel.from_pretrained on a
  native-mode adapter; fix or correct the claim + point to rekey. DONE: a test loads via stock PEFT, or
  README claim corrected + rekey path documented. WHERE: tests/ + README. EFFORT: M. GPU: small. DEPENDS: -.

## Epic E -- Observability + ETA
- **E1 ETA from --dry-run.** WHAT: dry-run estimates wall-clock (updates x measured per-step) + peak mem.
  DONE: dry-run prints "~Xh, peak ~Y GB" for the configured run. WHERE: train_nvfp4_lora.py dry-run path.
  EFFORT: S. GPU: yes (dry-run already loads). DEPENDS: -.
- **E3 per-phase timing in metrics.jsonl.** WHAT: extend the shipped metrics stream with load, forward,
  backward, optimizer, eval, checkpoint, and save timing. DONE: metrics.jsonl carries phase timings in
  addition to the shipped throughput/memory/ETA fields. WHERE: train_nvfp4_lora.py. EFFORT: S-M. GPU: no
  (passive). DEPENDS: shipped baseline metrics.

## Epic F -- Retention / forgetting eval
- **F1 general-ability held-out eval.** WHAT: small fixed general probe set, base vs FT, report
  degradation alongside task lift. DONE: a retention number in the eval bundle. WHERE: new eval + probe
  set. EFFORT: M. GPU: yes (serve). DEPENDS: -.

## Epic G -- Multi-task eval wiring (1.5; generators already exist)
- **G1 one-command before/after for GSM8K/HumanEval/BigCodeBench.** WHAT: wire existing generators into
  repro-style base-vs-adapter. DONE: repro_<task>.sh (or --task) produces base-vs-adapter per task.
  WHERE: scripts/ (eval_gsm8k.py, gen_humaneval.py, gen_bigcodebench.py exist). EFFORT: M. GPU: yes. DEPENDS: -.

## Epic H -- Quant-quality standing gate (1.5)
- **H1 broaden the "quant tax negligible" claim.** WHAT: quant-vs-bf16 adapter-quality on >1 model/dataset;
  standing gate. DONE: documented multi-model quant-tax table. WHERE: docs + eval. EFFORT: L. GPU: heavy. DEPENDS: -.

## Quick-win batch (CPU-only or tiny)
A1, A4, A6, B3, C1, C2, D3(test scaffolding), and E3 phase timing. The v1.5 refresh already shipped
error bars, per-db breakdown, the metric-divergence guard, run metadata, baseline metrics, and the
runtime apply-check; the next quick win is surfacing those checks in docs plus adding the data doctor.

## Sequencing
1. B (parity + determinism) -- gate for all later change.   2. A + G (eval credibility) -- parallel, no kernels.
3. C + D + E (data-check, provenance, observability).        4. F (retention).   5. H (quant gate).
Throughput / expert-serving stay AFTER these and gated on the value experiment.
