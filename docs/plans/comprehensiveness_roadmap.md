# Comprehensiveness roadmap: NVFP4 LoRA FT on DGX Spark (GB10)

What would make nybbloris / nvfp4-lora-spark a genuinely *complete* NVFP4 (Blackwell 4-bit)
LoRA fine-tuning + runtime-LoRA serving story on single/dual DGX Spark (GB10, sm_121).

## Thesis
The moat is **"fine-tune a huge NVFP4 model on Spark and TRUST THE ARTIFACT"** -- not more
kernels or families. The project should prioritize the layer where real users actually fail:
data correctness, eval honesty, reproducibility, and adapter portability. Correction:
**credibility infrastructure first, speed second.**
Key data point: the README already reports the Triton dequant took the 119B MoE step
984s -> 92s (10.7x) and "dequant is no longer the bottleneck" -- so throughput is incremental,
not the lead, and sequence packing is a numeric hazard that needs the parity gate to exist first.

## Context / current state
- Train: unified trainer auto-detects family + LoRA mode (native NVFP4 / FP8 / BF16 co-train),
  crash-safe checkpoint/resume, and writes run metadata plus JSONL metrics (config snapshot,
  git SHA, dataset hashes, package versions, supervised tokens/s, updates/s, CUDA/host memory,
  ETA). Dataset masking is mature (multi-turn assistant-only masking, truncation-drop, NaN guard
  -- all tested) but NOT surfaced to users as a data-check command.
- Families: Llama/Qwen3 dense, Qwen3.5/Mistral4 MoE, GLM-4.5-Air (fused-3D MoE), Mistral-Small, Nemotron.
- Targets: attention, dense MLP, routed experts (fused-3D) -- expert-LoRA train+serve PROVEN on one box
  through the emulation backend, including Qwen3.5-122B expert-LoRA runtime serving; VALUE EXPERIMENT
  IN FLIGHT (does expert-LoRA beat attention/shared-expert?).
- Serve trust: the binding contract is wrapped-model, format, and backend aware, and
  `scripts/serve_apply_check.py` verifies runtime application with a prompt-logprob delta so a
  loaded-but-no-op adapter cannot pass silently.
- Eval: Spider (deterministic gold-NLL + exact-set-match) now emits paired bootstrap CIs for EM/NLL
  deltas, per-db slices when `db_id` is present, empty-generation/no-op warnings, and metric-divergence
  warnings. GSM8K/HumanEval/BigCodeBench generators exist but are not wired into one-command
  before/after eval. Contamination and held-out-db checks are still missing.
- MEASURED throughput: ~82s/update on GLM-4.5-Air len1024 on one GB10 (~20h/full epoch; we cap to 2000ex).

## TIER 1 -- credibility infrastructure (the moat)
1. **Eval honesty / leakage (THE biggest gap).** DONE in v1.5: paired bootstrap CIs on EM/NLL deltas,
   per-slice/per-db breakdown, no-op/empty-generation warnings, and an "NLL improved but generation
   regressed" detector. Remaining: contamination/overlap check (n-gram + exact-question train<->eval)
   and held-out-db generalization split so one easy db can't carry the delta.
   Spider has only 140 train dbs vs 20 val dbs with identically-templated schemas -> the +15.3pp is
   contestable without this. Defends every headline number; unblocks the v1 traction gate.
2. **Train<->serve numerical parity + same-seed determinism as a CI/GPU gate.** DONE in v1.5:
   wrapped/format/backend-aware binding inspection and the runtime logprob-delta apply-check. Remaining:
   train-side delta == serve-side applied as an automated GPU gate per target type incl. experts; plus
   same-seed -> same-adapter / same-eval -> same-numbers. The trust backbone; MUST precede throughput
   changes (packing silently breaks numerics).
3. **`nybbloris data-check` / data doctor.** Template-render preview, assistant-mask coverage,
   TRUNCATION-DROP COUNT (silently dropping 15% of rows on a 20h run = invisible catastrophe),
   token-length histogram, packed-utilization estimate, empty-assistant-span flag, tokenizer/template/
   model hashes. The masking logic is already good; it just isn't surfaced.
4. **Adapter provenance + portability.** Manifest stamped into every adapter/merged dir: base repo+rev+
   safetensors hashes, tokenizer hash, chat-template hash, quant layout version, family-registry version,
   target coverage, rank/alpha/dropout, train git SHA, dep versions, masking/packing settings, eval
   summary, merge-compat status. Format versioning + compat policy. FIX the untested
   peft.PeftModel.from_pretrained round-trip (README:344) so the "standard PEFT adapter" claim is true.
5. **Observability + ETA.** DONE in v1.5: structured run metadata and JSONL metrics are written by the
   trainer. Remaining: wall-clock + memory ESTIMATE from --dry-run ("~31h, needs Y GB") and per-phase
   timing in the metrics stream.
6. **Retention / forgetting eval.** Small general held-out set, base vs FT -- "did FT damage general
   ability." (The other half of the original #4; leakage half above is more urgent.)

## TIER 1.5 -- promoted, low-effort / high-ROI
7. **Multi-task eval matrix.** Wire the EXISTING GSM8K/HumanEval/BigCodeBench generators into the
   one-command before/after. Kills "all your evidence is one task (Spider)." Nearly built.
8. **Quant-quality as a standing gate.** The load-bearing "quant tax on adapter quality is negligible"
   claim rests on ONE model/dataset (Nano-30B, 1081 rows). Broaden + make it a standing gate, not a footnote.

## TIER 2
9. **Throughput (now SAFELY behind the parity gate).** Sequence packing (numeric hazard w/ assistant
   masking + cross-doc attention -- do AFTER #2), grouped-MoE GEMM GPU validation (built, CPU-parity-tested),
   grad-checkpoint tuning. Incremental, not the lead (dequant already 10.7x'd).
10. **Target ablations + "what to target" guide** (shared-expert / embeddings / lm_head / attn-vs-MLP);
    gated on the value experiment. The current experiment is the seed.
11. **Instruction / chat-quality eval.** All bundled evals are auto-scorable extractive tasks; the common
    SFT-for-chat case has no gold string. Add an LLM-judge / preference harness OR explicitly scope out
    (today it silently overclaims generality).
12. **License-provenance stamp at merge.** merge_lora_into_nvfp4.py prints inherited base license +
    redistribution constraint into the output dir (merged checkpoints are derivative works).

## TIER 3 -- defer / cut on this hardware
13. **Fast expert-LoRA serving (marlin 2-box / faster emulation) -- GATED on the value experiment.**
    Do NOT build until experts demonstrably beat attention/shared-expert LoRA. High kernel/maintenance cost.
14. Native FP4 GEMM -- research; not the next bottleneck after packing/grouped-MoE.
15. More families -- after the gates above are mandatory + automated. Trustworthy-narrow > broad-shallow.
16. DoRA / LoRA+ / rank sweeps -- until a reproducible failure plain LoRA can't solve.
17. 2-box FSDP/TP training -- lowest; GB10 ships single-GPU, single-box is the appeal.
18. Multi-adapter serving -- Tier 3 / maybe cut for now.

## Sequencing
1. Finish parity + determinism gate (#2) -- cannot trust any later change without it.
2. Finish eval-credibility bundle (#1 contamination/held-out-db + #7 multi-task) in parallel -- no kernels,
   unblocks traction.
3. Adapter portability + provenance (#4) + data-check (#3) + observability/ETA (#5).
4. THEN throughput (#9), now guarded by the parity gate.
5. Target ablations (#10) -> fast expert-serving (#13) ONLY if the value experiment says experts win.
6. More families (#15) after the gates are automated.

NET: leverage is **credibility infrastructure (parity, leakage, data-check, provenance) THEN speed**.
The model already trains fast enough to be usable; the remaining work is to prove each result is real,
reproducible, and portable.
