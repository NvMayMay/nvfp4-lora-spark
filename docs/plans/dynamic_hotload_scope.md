# Dynamic hot-load LoRA: Spider repro + serve scope

Implementation scope for adding **runtime hot-load** LoRA (POST `/v1/load_lora_adapter`)
to the Spider reproduction + serve path. Today both `scripts/repro_spider.sh` and
`serve/run_glm45_air_nvfp4_*lora.sh` attach adapters at **launch time** via
`--lora-modules name=path`. This adds a `DYNAMIC=1` mode that serves the bare base and
registers the adapter over HTTP after READY ("serve once, swap many").

Grounding (read 2026-06-28):
- `nybbloris/cli.py` already gates the endpoint: `--allow-runtime-lora-updates` sets
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, and the else-branch **pops** any inherited value
  so the endpoint is off unless explicitly opted in (cli.py:204-209). Host default is
  `127.0.0.1` (cli.py:374).
- `docs/plans/DYNAMIC_LORA_VALIDATION_RUNBOOK.md` step (d) has the proven curl flow for
  load/unload against vLLM 0.22.1.
- `scripts/eval_retention.py` hits served **model names** only (`--models base myft`,
  lines 241-242, `_post`/`gold_nll`/`gen_sql`). It needs **zero changes** as long as the
  served name `myft` exists by the time it runs.
- vLLM 0.22.1 (qwen-serve venv): the runtime endpoint builds a `LoRARequest` and calls
  `engine_client.add_lora()` (`entrypoints/openai/models/serving.py:177-190`) — the same
  engine add path the launch-time preload uses. MoE `set_lora_context` is driven from
  `FusedMoEWithLoRA.set_lora()` (`lora/layers/fused_moe.py:375-378`), which both paths
  reach via the model manager's `add_adapter`. So for **dense** Spider models the runtime
  path is the same math as launch attach.

Spider demo models are DENSE (Llama-3.1-8B, Qwen3-32B, Mistral-24B): no fused-MoE LoRA
gate, no patch, runtime hot-load should "just work" under the existing flag.

---

## D1 — `DYNAMIC=1` mode for `repro_spider.sh` (CPU / code)

**Effort: S (~1h). Risk: low. Files: `scripts/repro_spider.sh` only.**

Add an env switch `DYNAMIC=${DYNAMIC:-0}`. When set, step 4 serves the base with
`--enable-lora` + runtime updates but **without** `--lora-modules`; after READY, register
the adapter via curl; step 5 eval is unchanged; teardown unloads.

### Shell sketch (step 4, the `if [ "$DYNAMIC" = "1" ]` branch)

```bash
ADAPTER_NAME=${ADAPTER_NAME:-myft}

if [ "$DYNAMIC" = "1" ]; then
  echo "DYNAMIC=1: serving bare base, hot-loading '$ADAPTER_NAME' after READY"
  MAX_JOBS=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 "$VLLM" serve "$MODEL_DIR" \
    --served-model-name base --host 127.0.0.1 --port "$PORT" \
    --enable-lora --max-lora-rank 32 --max-loras 2 \
    --max-model-len 8192 --enforce-eager "${EXTRA_ARGS[@]}" \
    --gpu-memory-utilization 0.6 --kv-cache-dtype fp8 > "$SERVE_LOG" 2>&1 &
else
  # ... existing launch-time --lora-modules myft=$ADAPTER_DIR/best path, unchanged ...
fi
SERVE_PID=$!
```

`cleanup()` gains a best-effort unload before the kill (idempotent, ignore errors):

```bash
cleanup(){
  if [ "$DYNAMIC" = "1" ]; then
    curl -s -X POST "http://127.0.0.1:$PORT/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' \
      -d "{\"lora_name\":\"$ADAPTER_NAME\"}" >/dev/null 2>&1 || true
  fi
  kill "$SERVE_PID" 2>/dev/null || true
  pkill -9 EngineCor 2>/dev/null || true
}
trap cleanup EXIT
```

After the existing READY wait loop, register the adapter (only in DYNAMIC mode):

```bash
if [ "$DYNAMIC" = "1" ]; then
  step "4b/5 hot-load adapter via /v1/load_lora_adapter"
  LOAD_RESP=$(curl -s -w '\n%{http_code}' -X POST \
    "http://127.0.0.1:$PORT/v1/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$ADAPTER_NAME\",\"lora_path\":\"$ADAPTER_DIR/best\"}")
  CODE=$(echo "$LOAD_RESP" | tail -n1)
  [ "$CODE" = "200" ] || { echo "load_lora_adapter failed ($CODE):"; echo "$LOAD_RESP"; exit 1; }
  # gate on the served name actually appearing
  curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$ADAPTER_NAME\"" \
    || { echo "ERROR: $ADAPTER_NAME not in /v1/models after load"; exit 1; }
  echo "hot-loaded $ADAPTER_NAME"
fi
```

Step 5 eval is the existing call, with `myft` -> `$ADAPTER_NAME`:

```bash
"$PYTHON" scripts/eval_retention.py --base-url "http://127.0.0.1:$PORT" \
  --dev-file "$DATA_DIR/spider.dev.chat.jsonl" \
  --models base "$ADAPTER_NAME" --n "$N" --out "$OUT"
```

Note: `eval_retention.py` needs **no change** — it only references model names. The
`--max-lora-rank 32` must still match the trained rank (r=32 in this repro).

---

## D2 — GPU validation on a dense model (Llama-3.1-8B)

**Effort: S (~30min GPU once D1 lands). Risk: low. Files: none (manual / runbook).**

Prove the runtime path is correct and equivalent to launch attach. Sequence:

1. `DYNAMIC=1 bash scripts/repro_spider.sh` — serve bare base.
2. Before load: `GET /v1/models` lists **`base` only**; a completion to `myft` returns
   404 / "model not found". **Assert:** myft absent pre-load.
3. `POST /v1/load_lora_adapter` (myft). **Assert:** HTTP 200; `/v1/models` now lists `myft`.
4. Behavioral divergence: same prompt to `base` vs `myft` differs (reuse the cli.py
   `_verify` divergence proxy, or `scripts/distinguish_ft.py`). **Assert:** `myft != base`.
5. `POST /v1/unload_lora_adapter` (myft). **Assert:** HTTP 200; `/v1/models` drops `myft`;
   a query to `myft` now 404s.
6. **Equivalence assert (the load-bearing one):** run `eval_retention.py` once in DYNAMIC
   mode and once in launch-attach mode (same `N`, same dev file, deterministic NLL +
   greedy EM). Hot-loaded `mean_gold_nll[myft]` and `exact_set_match[myft]` must match the
   launch-time numbers within numerical tolerance (NLL ~exact since teacher-forced; EM
   exact-or-within-1-row from greedy non-determinism). This proves runtime add == preload.

Existing baseline files to diff against: `spider_retention_llama8b_e2.json`.

---

## D3 — Payoff demo: serve once, swap many

**Effort: S (~30min, builds on D2). Risk: low. Files: optional thin demo script or
README snippet; no core change.**

With one running base server (`--max-loras 2`):

1. `POST load` adapter A (`myft`) and adapter B (a 2nd Spider adapter, e.g. a different
   epoch / rank dir). **Assert:** `/v1/models` lists `base, myft, ftB`.
2. Query A, B, and base — all three respond; A and B each diverge from base and from
   each other. **Assert:** three distinct behaviors, one server, zero restarts.
3. `POST unload` A. **Assert:** A 404s; B still served; `base` byte-for-byte unchanged.

This is the "serve once, swap many" headline. Keep it as a documented curl sequence
(append to `REPRODUCE_SPIDER.md` or a `serve/README.md` section); a standalone script is
optional. Note `--max-loras` caps co-active adapters (default 2 here → A+B fills it;
raise it for >2 concurrent).

---

## D4 — Security (must hold for every dynamic path)

**Effort: 0 (already enforced); risk: HIGH if regressed. Files: none, document only.**

`/v1/load_lora_adapter` loads an **arbitrary filesystem path** from the request body
into the server. Non-negotiable invariants:

- **Bind 127.0.0.1 only.** repro_spider.sh already passes `--host 127.0.0.1`; never
  default the dynamic serve to `0.0.0.0`. Same for the cli.py default (cli.py:374).
- **Off by default.** The endpoint is live only with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`.
  cli.py sets it solely under `--allow-runtime-lora-updates` and otherwise **pops** any
  inherited value (cli.py:204-209). In repro_spider.sh, set the env var **inline on the
  vllm invocation** (as in the D1 sketch) so it cannot leak to other processes, and only
  under `DYNAMIC=1`.
- **Document the threat** in the new mode's header comment: a network-reachable server
  with this flag on = arbitrary local-path load = remote-triggered code/weight load.
  Localhost + opt-in is the entire blast-radius control.

---

## D5 — MoE caveat (open question, test — do not assume)

**Effort: M (GPU, gated on a free 122B/106B slot). Risk: medium-unknown.**

For MoE NVFP4 (GLM-4.5-Air, Qwen3.5-122B) two LoRA target classes exist:

- **Attention-only** over CUTLASS MoE: handled by
  `serve/vllm_patches/attention_only_lora_cutlass_moe.py` (pins every FusedMoE
  LoRA-disabled, rejects expert-targeting adapters). Runtime hot-load of an
  **attention-only** adapter here is already exercised in the runbook step (d) and gated
  by the patch — lower risk.
- **Expert** adapters: must go through the **emulation / marlin** MoE path
  (`--moe-backend marlin`, `serve/vllm_patches/marlin_repack_patch.py`), validated so far
  only for **launch-time** attach.

**Open question to TEST (not assume):** does the runtime load path compose with the
expert path the same way launch-time attach does — i.e. is `FusedMoEWithLoRA.set_lora()`
→ `set_lora_context()` (`lora/layers/fused_moe.py:375-378`,
`fused_moe/experts/lora_experts_mixin.py:28,58-99`) invoked on a **runtime-loaded**
adapter identically to a preloaded one?

Static read says it *should*: the runtime endpoint funnels through
`engine_client.add_lora()` (serving.py:190) into the same model-manager `add_adapter`
that launch-time preload uses, and `set_lora` is per-wrapped-layer at add time, not a
launch-only step. **But** this is an assertion to confirm on GPU, because the
emulation/marlin expert path was only ever driven by the preload and may rely on
load-order or a one-shot stacked-buffer build that the incremental runtime add does not
reproduce.

Validation (when a MoE slot is free): serve a marlin/emulation base with
`--enable-lora` + runtime updates and **no** `--lora-modules`; `POST load` an
**expert-targeting** adapter; assert HTTP 200, served name appears, and behavior diverges
from base (distinguish_ft compare > 0 differing domain prompts). If the runtime load 200s
but output == base, `set_lora_context` did not fire on the runtime add → fall back to
launch-time attach (or merge-then-serve) for expert MoE, and file the gap. **Do not claim
runtime expert hot-load until this passes.**

---

## Summary table

| Item | What | Files | Effort | Risk |
|------|------|-------|--------|------|
| D1 | `DYNAMIC=1` serve-bare + curl load/unload in repro_spider.sh | `scripts/repro_spider.sh` | S (~1h) | low |
| D2 | GPU validate on Llama-8B: pre-absent → load → diverge → unload → gone; numbers == launch attach | none (manual) | S | low |
| D3 | serve once, swap many: 2 adapters, query both, unload one, base unchanged | `REPRODUCE_SPIDER.md`/`serve/README.md` (+ optional script) | S | low |
| D4 | 127.0.0.1 + off-by-default; document threat | none (already enforced) | 0 | high if regressed |
| D5 | does runtime load compose with `--moe-backend` emulation/marlin expert path? test, don't assume | none until proven; possibly new MoE dynamic serve script | M (GPU-gated) | medium-unknown |

**Net:** D1–D4 are a tight dense-model win on existing infra (one shell file edited, one
GPU validation, eval untouched). D5 is the genuinely open, GPU-gated MoE question and is
explicitly an experiment, not a claim.
