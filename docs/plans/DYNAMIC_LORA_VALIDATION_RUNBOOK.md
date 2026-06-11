# Dynamic attention-only LoRA over CUTLASS NVFP4 MoE: tonight GPU runbook

Mechanical validation sequence for the patch in
`serve/vllm_patches/attention_only_lora_cutlass_moe.py` (Qwen3.5-122B-A10B CT
NVFP4, vLLM 0.22.1, DGX Spark GB10 sm_121).

This runbook was produced by an independent re-verification pass on 2026-06-10.
The CPU-side checks (A-E below) all passed against the real install and the real
`checkpoint_step_50` adapter. What remains can only be settled on the GPU, which
is owned by the 13h training run until ~01:30. Do not start any step until the
trainer has exited and `nvidia-smi` shows the device free.

Independent confidence after CPU verification: ~82 percent that the patched
serve comes up and applies the adapter on the first GPU attempt. The residual
~18 percent is GPU-only and is enumerated under "Residual risks" at the end.

SECOND-PASS DE-RISK UPDATE (2026-06-10, static analysis + CPU-executable tests,
CUDA_VISIBLE_DEVICES=""): three of the four residual risks the first pass marked
GPU-only have been shrunk. Updated confidence ~90 percent. See the rewritten
"Residual risks" table at the end; the headline result is that vLLM's OWN
adapter-load stack accepts our real adapter end-to-end on CPU (everything except
the final CUDA placement), and one prior assumption (GDN wrappers never execute
punica) was found WRONG and corrected, but the corrected shapes all clear the
kernel constraints.

---------------------------------------------------------------------------
## 0. Pre-flight (run BEFORE touching the GPU)

```bash
# Confirm the trainer is done and the device is free.
nvidia-smi
tail -n 5 /home/veritan-spark-01/Veritan/Sandbox/adapters/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5/metrics.jsonl
# ABORT if any python training process still holds GPU memory.

# Re-run the CPU verification suite (no CUDA, ~30s). All six must PASS.
export CUDA_VISIBLE_DEVICES=""
VENV=/home/veritan-spark-01/Veritan/.venvs/qwen-serve
PATCHDIR=/home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark/serve/vllm_patches
cd /tmp/patch_verify
for t in test_A_patch_apply.py test_B_key_remap.py test_C_expert_guard.py \
         test_D_sitecustomize.py test_D2_from_lora_tensors_wrapper.py \
         test_E_wrap_audit.py; do
  PYTHONPATH=$PATCHDIR $VENV/bin/python $t >/tmp/patch_verify/out_$t.log 2>&1 \
    && echo "PASS $t" || echo "FAIL $t -> out_$t.log"
done
unset CUDA_VISIBLE_DEVICES   # release the CPU-only guard before serving
```
ABORT criteria for step 0: any test FAILs, or the GPU is not free.

Pick the latest checkpoint at run time (today only `checkpoint_step_50` exists):
```bash
ADAPTER_ROOT=/home/veritan-spark-01/Veritan/Sandbox/adapters/qwen3_5_122b_a10b_rh_nvfp4_lora_ich_v3_5
LATEST_CKPT=$(ls -d $ADAPTER_ROOT/checkpoint_step_* 2>/dev/null \
  | grep -v '\.tmp' | sort -t_ -k3 -n | tail -1)
echo "latest checkpoint: $LATEST_CKPT"
```

---------------------------------------------------------------------------
## (a) Unpatched base CUTLASS serve sanity (bisects the Jun 8 0.21 -> 0.22.1 upgrade)

Serve the SAME RedHatAI base the dynamic launcher uses, with NO LoRA args and
NO patch. `run_qwen35_122b_nvfp4.sh` defaults MODEL_DIR to the non-RedHatAI dir,
so override it.

```bash
MODEL_DIR=/home/veritan-spark-01/Veritan/Models/RedHatAI-Qwen3.5-122B-A10B-NVFP4 \
  ./serve/run_qwen35_122b_nvfp4.sh
```
EXPECT in the engine log:
- `Using 'VLLM_CUTLASS' NvFp4 MoE backend`
- server reaches `Application startup complete` / `Uvicorn running on ... :8000`
- NO `sitecustomize ... attention_only_lora_cutlass_moe` line (patch not active here)

Smoke it (text completion; the distinguish script uses /v1/completions):
```bash
curl -s http://localhost:8000/v1/models | python -m json.tool   # lists qwen3.5-122b-a10b-nvfp4
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-122b-a10b-nvfp4","prompt":"The principles of GCP compliance include","max_tokens":64,"temperature":0}' \
  | python -m json.tool
```
EXPECT: a coherent completion; generation in the 11.8-14.5 tok/s band (watch the
engine throughput log line).

ABORT criteria: backend is not VLLM_CUTLASS, OOM at load, or no coherent output.
If THIS step fails, the Jun 8 vLLM upgrade broke the base serve; that is a
pre-existing problem, NOT the patch. Stop and fix the base before going further.

Stop the server (Ctrl-C) before step (b); only one 122B server fits in 131 GB.

---------------------------------------------------------------------------
## (b) Patched serve + adapter load

```bash
./serve/run_qwen35_122b_rh_ct_dynamic_lora.sh
```
EXPECT, in order:
- ONE per process (APIServer AND the spawned EngineCore), on stderr:
  `[sitecustomize pid=...] applied attention_only_lora_cutlass_moe patch`
  (you should see it at least twice, once per pid)
- `attention_only_lora_cutlass_moe: FusedMoEConfig.is_lora_enabled pinned to False`
- `Using 'VLLM_CUTLASS' NvFp4 MoE backend`   <- the gate that stock vLLM would
  have turned into a raise; if you see VLLM_CUTLASS here the patch worked.
- `attention_only_lora_cutlass_moe: excluding FusedMoE module suffixes ['experts'] ...`
- `attention_only_lora_cutlass_moe: remapped 96 flat-layout adapter keys ...`
- `Application startup complete`

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
# EXPECT both: qwen3.5-122b-a10b-nvfp4  AND  ich_v3_5
```
Smoke both model names:
```bash
for M in qwen3.5-122b-a10b-nvfp4 ich_v3_5; do
  echo "=== $M ==="
  curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"prompt\":\"The primary objective of ICH E6(R2) is to\",\"max_tokens\":64,\"temperature\":0}" \
    | python -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])'
done
```
EXPECT: both return coherent text; the `ich_v3_5` output should read differently
on the ICH-domain prompt.

ABORT criteria:
- backend is NOT VLLM_CUTLASS -> piece A failed; do NOT proceed, roll back.
- startup raises in `_return_or_raise` / `select_nvfp4_moe_backend` with
  "...does not support ... LoRA" -> piece A did not take in EngineCore; check
  that the sitecustomize line printed in the EngineCore pid too.
- you do NOT see `remapped 96 flat-layout adapter keys` -> the adapter keys were
  not remapped; the adapter would silently serve the base. Stop.
- a startup/first-forward CUDA crash that mentions a GDN module
  (conv1d / in_proj_qkvz / in_proj_ba / out_proj) or a punica/Triton autotune
  error on a 16384-wide slice -> see Residual risks; roll back to merge path.

---------------------------------------------------------------------------
## (c) Distinguishing-prompt FT-vs-base check (no restart; both served at once)

`scripts/distinguish_ft.py` interface (verified): `collect --url --model
--output-jsonl [--max-tokens]` hits `/v1/completions` at temperature 0 over 100
fixed prompts (29 ICH/regulatory domain + 71 filler); `compare base.jsonl
ft.jsonl` reports identical vs differing counts.

```bash
VENV_PY=/home/veritan-spark-01/Veritan/.venvs/qwen-serve/bin/python
cd /home/veritan-spark-01/Veritan/Sandbox/repos/nvfp4-lora-spark

$VENV_PY scripts/distinguish_ft.py collect --url http://localhost:8000 \
    --model qwen3.5-122b-a10b-nvfp4 --output-jsonl /tmp/qwen_base.jsonl
$VENV_PY scripts/distinguish_ft.py collect --url http://localhost:8000 \
    --model ich_v3_5 --output-jsonl /tmp/qwen_ft.jsonl
$VENV_PY scripts/distinguish_ft.py compare /tmp/qwen_base.jsonl /tmp/qwen_ft.jsonl
```
EXPECT: several ICH-domain prompts diverge; most generic/filler prompts identical.

ABORT/diagnose: if ZERO prompts differ, the adapter is NOT applying. This is the
silent-no-op failure mode the patch is meant to prevent. Re-read the step (b)
`remapped 96 flat-layout adapter keys` line. If that line was present but output
is still identical, the punica path is not adding the LoRA product (escalate;
do not ship dynamic LoRA for the demo, fall back to merge path).

Control (optional, proves the harness): collect base twice and `compare` the two
base files -> EXPECT 0 differing (identical).

---------------------------------------------------------------------------
## (d) Adapter hot-swap via the runtime LoRA update endpoint

The launcher exports `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, so POST endpoints are
live. Swap in the latest checkpoint as a second adapter:

```bash
curl -s -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"ich_v3_5_step50\",\"lora_path\":\"$LATEST_CKPT\"}"
# EXPECT: HTTP 200, success body.
curl -s http://localhost:8000/v1/models | python -m json.tool   # now lists ich_v3_5_step50 too
```
Smoke the new adapter, then negative-test the guard (must raise, not silently
load) with any MoE/expert-targeting adapter you have on disk:
```bash
# NEGATIVE TEST (expect an error mentioning attention_only_lora_cutlass_moe and
# "merge-then-serve", NOT a 200):
curl -s -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"should_fail","lora_path":"/path/to/any/moe-targeting-adapter"}'
```
Unload when done:
```bash
curl -s -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' -d '{"lora_name":"ich_v3_5_step50"}'
```
ABORT criteria: a load that should fail (expert-targeting) returns 200 -> the
guard did not fire on the runtime path; stop and investigate before trusting any
adapter swap. Note `--max-loras 2`: with `ich_v3_5` already loaded you have room
for exactly one more co-active slot.

---------------------------------------------------------------------------
## (e) Throughput sanity vs merge-then-serve

Reuse the step (c) timings as a single-stream proxy (each collect prints per-
prompt latency; the May 31 baseline is 11.8-14.5 tok/s). For a concurrent number,
fire 4 parallel completions and eyeball aggregate tok/s.

Then run the merge path on the SAME prompts for comparison (separate server;
stop the dynamic server first):
```bash
# stop the dynamic server (Ctrl-C), then:
./serve/run_qwen35_122b_rh_ct_lora.sh merge    # bakes MERGED_DIR (minutes)
./serve/run_qwen35_122b_rh_ct_lora.sh serve    # serves merged checkpoint
$VENV_PY scripts/distinguish_ft.py collect --url http://localhost:8000 \
    --model qwen3.5-122b-a10b-nvfp4+ich_v3_5 --output-jsonl /tmp/qwen_merged.jsonl
```
EXPECT: base-model requests on the dynamic server match step (a); adapter
requests a few percent slower than merged (punica shrink/expand per wrapped
linear). Decision rule: if dynamic LoRA is >~15 percent slower than merged on
the 4-way concurrent number, use the merged checkpoint for the demo and keep the
dynamic path for iteration.

Cross-check correctness: `compare /tmp/qwen_ft.jsonl /tmp/qwen_merged.jsonl`
should be MOSTLY identical (dynamic adapter vs merged adapter are the same math;
small numerical drift is acceptable, large divergence is a red flag).

---------------------------------------------------------------------------
## Rollback (any step fails, or for the demo you want the safe path)

1. Stop the dynamic server (Ctrl-C).
2. Disable the patch for any future launch from this shell:
   ```bash
   unset VLLM_PATCH_ATTN_ONLY_LORA_CUTLASS_MOE
   ```
   The patch is inert without this gate var even though `PYTHONPATH` still
   points at `serve/vllm_patches` (the marlin patch stays on, harmless). The
   env-var gate is the entire blast-radius control; do not delete the patch file.
3. Fall back to merge-then-serve (proven Path B):
   ```bash
   ./serve/run_qwen35_122b_rh_ct_lora.sh merge
   ./serve/run_qwen35_122b_rh_ct_lora.sh serve
   ```
   Served name `qwen3.5-122b-a10b-nvfp4+ich_v3_5`, port 8000, VLLM_CUTLASS, no
   LoRA wiring at all. This is the lowest-risk path for the demo.

---------------------------------------------------------------------------
## Residual risks (after 2026-06-10 second-pass de-risk)

Status legend: RESOLVED (settled by static analysis or CPU test), REDUCED
(materially de-risked, narrow GPU-only remainder), GPU-ONLY (irreducible without
the device).

### Per-module punica shape audit (all shapes enumerated; TP=1, r=16, max_loras=2)

All dims computed from the model config and CONFIRMED against the real adapter
safetensors (q lora_B=(16384,16), k/v=(512,16), o lora_A=(16,8192)). vLLM verb:
shrink does x[hidden]->[rank] (K=hidden, N=rank), expand does [rank]->[out]
(K=rank, N=out). Kernels mask N via `offset_n % N` + `c_mask` (kernel_utils.py
:222-229, 325-333); NO power-of-2 requirement on N, NO max-N guard. Default
kernel configs are used (no VLLM_TUNED_CONFIG_FOLDER), so there is NO autotune
key-space to miss and NO tl.constexpr baking in a width limit
(triton_ops/utils.py:218-271). Index/metadata tensors are int32 but bounded by
max_num_batched_tokens=16384 and max_loras=2; LoRA-weight data pointers are
uint64 (utils.py:53,112). No int32 overflow at any of these sizes.

  | module        | wrapper                                   | in  | out (slices)        | proven? |
  |---------------|-------------------------------------------|-----|---------------------|---------|
  | qkv_proj      | MergedQKVParallelLinearWithLoRA (3 slice) | 3072| q16384,k512,v512    | NOVEL   |
  | o_proj        | RowParallelLinearWithLoRA (1 slice)       | 8192| 3072                | NOVEL   |
  | in_proj_qkvz  | MergedColumnParallelLinearVariableSlice   | 3072| 2048,2048,8192,8192 | NOVEL   |
  | in_proj_ba    | MergedColumnParallelLinearWithLoRA (2 sl) | 3072| 64,64               | NOVEL   |
  | out_proj (GDN)| RowParallelLinearWithLoRA (1 slice)       | 8192| 3072                | NOVEL   |
  | conv1d (GDN)  | ColumnParallelLinearWithLoRA              | 4   | 12288               | NOT RUN |

  Field-proven precedent: the Nano-30B punica runs on this box used adapters
  targeting `up_proj`/`down_proj` at **r=8** (adapter configs under
  Sandbox/adapters/nemotron_3_nano_nvfp4_*). They prove the GPU + punica path on
  sm_121 in general, but they do NOT cover our attention/GDN shapes or r=16. So
  every shape above is "novel-but-within-constraints": each clears the static
  constraints (masking handles non-multiples of BLOCK_N; q=16384 and qkvz=20480
  are far below any int32 ceiling), but none is byte-for-byte field-proven.
  Verdict: novel-but-within-constraints for all running shapes; conv1d never
  reaches a kernel (see risk 2). A failure would be LOUD at first forward, not
  silent. -> REDUCED to "first-forward smoke" (step b/c).

### 1. Punica/Triton on the 16384-wide fused-q slice -> REDUCED
   Confirmed the q slice is num_heads*head_dim*(1+attn_output_gate)=32*256*2=
   16384 (qwen3_next.py:245 `total_num_heads*(1+attn_output_gate)`, attn_output_
   gate=true in config; adapter lora_B row count 16384 confirms). The expand
   kernel handles 16384 via cdiv+mask, default config block_n=64 (num_slices>1),
   no autotune. No static blocker found. Remaining GPU-only sliver: actual Triton
   JIT/launch on sm_121 at this width. LOUD if it fails.

### 2. GDN linears under LoRA -> CORRECTED then REDUCED  (prior pass was WRONG here)
   target_modules defaults to None at serve time (LoRAConfig.target_modules=None,
   config/lora.py:48; launcher passes no --lora-target-modules), so
   is_in_target_modules returns True for EVERY supported linear (lora/utils.py
   :282) and `_match_target_modules` wraps all of them (model_manager.py:389).
   GDN modules ARE wrapped: in_proj_qkvz -> MergedColumnParallelLinearVariable
   SliceWithLoRA (4 slices), in_proj_ba -> MergedColumnParallelLinearWithLoRA
   (2 slices), out_proj -> RowParallelLinearWithLoRA, conv1d ->
   ColumnParallelLinearWithLoRA.
   CORRECTION: the prior pass claimed "the wrapper's forward never runs" for GDN.
   That is true ONLY for conv1d, which is read as a raw `.weight` tensor
   (qwen_gdn_linear_attn.py:1326 `self.conv1d.weight.view(...)`), so its wrapper
   never executes a kernel. But forward_cuda DOES call the modules
   `self.in_proj_qkvz(...)`, `self.in_proj_ba(...)` (lines 923-924) and out_proj
   via `_output_projection` (line 968 -> 869). Those wrapped calls go through
   base_linear.py `_apply_lora_to_output` -> `add_lora_linear` -> add_shrink +
   add_expand UNCONDITIONALLY (punica_gpu.py:203-264). There is NO per-module
   "skip if buffers all-zero" anywhere; the only skips are (a) the whole-batch
   `no_lora_flag_cpu` early-exit (kernel_metadata.py:121-126) and (b) the per-CTA
   `lora_id == -1` early-exit. For an active-adapter request the GDN kernels DO
   launch over the zeroed buffers (numerically a no-op: shrink/expand of all-zero
   lora_a/lora_b adds 0).
   Net effect: GDN shapes are now folded into the audit above and must clear the
   kernel constraints, which they do (the 4-slice variable-width path is the SAME
   GQA `same_stride=False` mechanism the real qkv uses, expand_op.py:78-81; the
   in_proj_ba out=64 is tiny; out_proj is identical shape to o_proj). So this is
   no longer "very likely a no-op we never run" but "a real kernel launch we have
   statically cleared". GPU-only remainder: same first-forward smoke as risk 1.
   Wasted compute is small (12 linear-attn-heavy... actually 36 GDN layers x
   ~20k-wide expand over zeros); a perf note, not a correctness risk.

### 3. Base-serve 0.21 -> 0.22.1 default drift -> RESOLVED (pin-list added)
   Audited the 0.22.1 defaults that touch the proven CUTLASS recipe and pinned
   the load-bearing ones in run_qwen35_122b_rh_ct_dynamic_lora.sh:
     * --moe-backend cutlass: STILL VALID. map_nvfp4_backend "cutlass" ->
       VLLM_CUTLASS (oracle/nvfp4.py:144); explicit-backend path intact
       (nvfp4.py:236-258). New swiglu_limit guard (nvfp4.py:246-255) does NOT
       fire: FusedMoEConfig.swiglu_limit defaults None (config.py:1279) and the
       Qwen3.5 config does not set it.
     * --enforce-eager: REQUIRED on Spark (graph capture OOM, serve/README.md);
       already set. Keeps specialize_active_lora=False path (lora.py:67), no
       cudagraph LoRA-count specialization.
     * --no-enable-prefix-caching: PINNED. 0.22.1 default flipped to True
       (cache.py:91); for this hybrid GDN model it would engage the mamba
       prefix-cache path the 0.21 base serve never used. Pinned OFF.
     * --enable-chunked-prefill: PINNED ON. 0.22.1 resolves it from
       is_chunked_prefill_supported=True (model.py:1817); pinned ON so a future
       default flip can't trip verify_max_model_len (scheduler.py:260) or the
       "disabling may crash" warning (arg_utils.py:2383).
     * mamba-cache-mode: left at default "none" (cache.py:132). Model only
       rejects "all" (qwen3_5.py:459); "align" adds block-size constraints
       (vllm.py:2101) not needed at this concurrency.
   Step (a) still bisects any base regression empirically, but the config-drift
   surface is now pinned.

### 4. Property-override of the dataclass field -> RESOLVED (re-confirmed)
   Re-confirmed on this exact install: the dataclass __init__/layer.py:334 write
   is swallowed and reads return False. Standing caveat: re-audit on ANY vLLM
   upgrade (same class as the peft 0.19.1 in-place patch). Not GPU-dependent.

### 4b/Task-3 result (highest value): vLLM's OWN load path accepts our adapter -> RESOLVED on CPU
   Ran, with the patch applied and CUDA_VISIBLE_DEVICES="", the real load stack
   against checkpoint_step_100:
     * PEFTHelper.from_local_dir parsed adapter_config.json: r=16, lora_alpha=32,
       scaling=2.0, target_modules=[q,k,v,o], bias none, use_dora False. Our 10
       keys map to PEFTHelper's 3 required fields (r, lora_alpha, target_modules)
       + optional ones; the 4 unknown keys (peft_type, task_type, lora_dropout,
       inference_mode, fan_in_fan_out, base_model_name_or_path) are silently
       dropped by from_dict's field filter (peft_helper.py:77). validate_legal
       against LoRAConfig(max_lora_rank=16) PASSED. Suffix-list target_modules is
       accepted (no rejection).
     * LoRAModel.from_local_checkpoint(device="cpu") completed for BOTH
       dtype=bfloat16 AND float32: 96 tensors -> 48 LoRA modules, ALL 48 remapped
       to language_model.model.layers.* (0 still-flat, so NO silent no-op on
       GPU), ZERO landing on GDN/experts. check_unexpected_modules passed (q/k/v/o
       suffixes accepted). Shapes verified: q lora_b=(16384,16), k/v=(512,16),
       o lora_a=(16,8192).
     * The ONLY thing that stopped before completion is the .pin_memory() / .to(
       "cuda") device placement (lora_model.py:157) which needs a CUDA runtime;
       with pin_memory stubbed off the assembly finishes cleanly. Everything that
       is LOGIC (parse, hf_to_vllm_mapper, patch guard+remap, dtype cast,
       LoRALayerWeights assembly) ran and passed. The negative path was also
       re-confirmed: an expert-targeting tensor dict RAISES the patch's
       merge-then-serve error; shared_expert/gate WARNS not raises.
   This is vLLM's own code accepting our adapter end-to-end minus GPU placement.

### 5. End-to-end numerical correctness of the dense LoRA product on quantized
   base -> GPU-ONLY (irreducible). Only observable by serving; step (c)/(e)
   distinguish/compare is the check.

### Irreducible GPU-only remainder (what truly cannot be settled on CPU)
   a. First Triton JIT + launch of the punica shrink/expand kernels on sm_121 at
      our novel widths (q=16384, in_proj_qkvz=20480, plus the small ones). LOUD
      if it fails (kernel error at first forward), caught by step (b)/(c).
   b. Numerical correctness of dense LoRA on the NVFP4-quantized base at runtime
      (risk 5) -> step (c)/(e).
   c. UMA memory headroom with the LoRA static buffers + the extra GDN wrapper
      buffers (now known to be allocated for all 36 GDN layers) on top of the
      120B base at gpu-memory-utilization 0.70; only measurable on the box.
   Everything else (load path, key remap, shape legality, config drift, property
   override) is now settled off-GPU.

Bugs / corrections found during second-pass de-risk:
   * Prior pass's claim that the GDN LoRA wrappers' forward "never runs" is
     WRONG for in_proj_qkvz / in_proj_ba / out_proj (only conv1d is exempt).
     Corrected above; the corrected kernels are statically cleared, so the
     conclusion (no crash expected) stands, but the reasoning is now accurate
     and the GDN shapes are part of the audited set.
   * No bug in the patch itself: all four pieces (A property, B discovery
     exclusion, C wrapper fence, D guard+remap) are wired correctly and
     idempotent; the remap maps exactly the 96 real adapter tensors onto the
     vLLM module tree with zero landing on GDN/expert layers, now verified by
     running vLLM's own loader on CPU rather than by inspection alone.
