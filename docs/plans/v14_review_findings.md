# v1.4 dual-review findings (Opus 4.8 max + gpt-5.5 high)

Two independent reviewers audited the released v1.4 tag (pristine worktree / inline bundle).
Both verdicts: **SHIP-WITH-FIXES**. Consensus: the engineering core is solid and well-tested
(binding-contract analyzer, trainer family/mode/MoE detection, atomic crash-safe save,
numerical-oracle + serve-contract tests all verified). The gaps cluster on the **Spider
headline's honesty** (defaults/docs/artifacts) plus a few correctness/doc reconciliations.
None are crashes.

## Ranked fix-list (consensus severity)

### P0 - headline credibility (both reviewers, top-3 for both)
1. **One-command defaults != headline.** `repro_spider.sh` defaults `N=200 EPOCHS=1`, but the
   table says "full 1034-row dev, 2 epochs". Fix: default `EPOCHS=2 N=1034`, OR relabel the
   table as the explicit-env full repro and mark the default a smoke. [Opus major / gpt major]
2. **No shipped result artifact.** Headline + cross-family numbers are prose-only; nothing in
   `results/` to check against. Fix: commit the actual `spider_retention.json` (+ cross-family)
   under `results/spider/` and link from the table. [both]
3. **REPRODUCE_SPIDER internal contradiction.** Intro/training say "one-epoch"; the table says
   "2 epochs"; the eval cmd shows `--n 200` under a "full 1034-row" header. Fix: make it
   internally consistent; tie each number to the exact command that produced it. [gpt major]

### P1 - correctness / honesty
4. **Eval prompt-format skew.** Training applies the chat template; `eval_retention.py` generates
   via raw `/v1/completions` (no template). Model scored off-distribution; for reasoning models
   it produces EMPTY output. CONFIRMED LIVE on GLM-4.5-Air (200/200 empty). [Opus major / gpt minor]
   -> FIXED (uncommitted): added `--chat` (server-side template) + `--thinking` (+ `_extract_sql`).
5. **EM 0/0 reported as 0.0, not null.** Weakens "refuses to fake a result"; a consumer reads
   `exact_set_match: 0.0` as real. Fix: null on `em_n==0`, mirroring NLL. [gpt major]
   -> FIXED (uncommitted): em_n==0 -> null; plus all-empty greedy now flagged as FAILURE.
6. **Stale "partial quantization = hard error" docs + dead flag.** README:309 + `decide_lora_mode`
   docstring claim a hard error needing `--allow-partial-targets`; code co-trains BF16 natively and
   the flag gates nothing (tests assert the new behavior). Fix: update docs/argparse to describe
   co-training, or wire the flag to a real check. [both major]
7. **Grad-accum drops final partial window.** Optimizer steps only at `micro_step % grad_accum==0`,
   so ~8 examples/epoch (7000 % 16) are silently dropped -> "full train set" slightly false +
   reproducibility drift. Fix: flush a final partial window at epoch end, or document drop_last.
   [gpt major]

### P2 - robustness / consistency
8. **`--resume-from` resets `best_val_loss=inf`** -> first post-resume eval can overwrite `best/`
   with a worse adapter. Fix: persist/restore `best_val_loss` in `train_state.pt`. [Opus minor]
9. **rekey transform duplicated vs `nybbloris.plan.REKEYS`**, no shared source, no PASS test.
   Fix: derive rekey from plan semantics OR add a test that `serve_plan()` returns PASS on the
   script's output. [gpt]
10. **README roadmap stale**: "Bundled eval harness - not shipped today" while v1.4 ships
    `eval_retention.py` as a headline path. Fix: remove/update. [gpt minor]
11. **Test count stale** ("79 tests" -> actually ~113). [Opus minor]
12. **`serve --verify` greedy non-determinism** vs "binding contract" language: keep static
    `inspect` as the contract, label `--verify` advisory. [gpt minor]
13. **VLM-detect grep too broad** (`*ForConditionalGeneration` etc.) - tighten to `vision_config`
    + explicit arch list. [Opus minor]
14. **READY wait = coarse log-grep**; reuse the CLI's `/v1/models` probe. [Opus minor]
15. **PEFT round-trip caveat** should point to `rekey_lora_for_vllm.py`. [Opus nit]
16. **No-op detection only on exact rounded equality** - can miss subtle no-ops. [gpt minor]

## Already fixed (uncommitted, eval_retention.py)
- #4 `--chat` server-side templating; `--thinking` + `_extract_sql` for reasoning models.
- #5 EM `em_n==0 -> null`; all-empty greedy flagged as measurement FAILURE (`empty_generations`).

## Verbatim reviews
- Opus: scratchpad transcript (this session).
- gpt-5.5: scratchpad/codex_v14_review.out
