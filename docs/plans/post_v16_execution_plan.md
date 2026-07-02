# Post-v1.6 execution plan (toward the north star)

**North star:** LoRA fine-tune *any* NVFP4 model up to ~120B on a single DGX Spark (GB10,
sm_121, ~128GB UMA), then serve that fine-tune with the LoRA provably applied at runtime.

**Status at start:** v1.6.0 on `main` (HEAD `92dab47`). Train half largely proven; serve-live
proven *correct* but slow on routed MoE (emulation backend ~0.7 tok/s vs 12-14 on CUTLASS);
"any model" blocked by an 8-entry family allowlist. Source of items: the v1.6 fleet synthesis +
the Fable independent review (2026-07-01).

This plan is written to be executed **maximally autonomously** once we commit to starting. The
autonomy model below is a first-class part of the plan: it defines exactly what I run unattended,
what I batch for a single approval, and the guardrails that let me not stop and ask.

---

## 1. Autonomy model (how this runs once we say "go")

### 1.1 Authorization I need at "go" (grant once, covers the whole run)
Granting these at start is what makes the run autonomous. Default standing rules stay in force
except where a grant explicitly widens them:

- **G1 - local commits on a dedicated branch.** Create `feat/post-v16` off `main` and make
  incremental **local** commits as checkpoints. Selective path staging only (see guardrail R1).
  Nothing is pushed under this grant.
- **G2 - GPU runs on the idle Sparks (Box A + Box B).** Launch training/serve/validation jobs,
  kill stray processes by PID, read `torch.cuda.mem_get_info()`, run background jobs.
- **G3 - artifact generation.** Write results under `results/`, reports under `docs/`, update the
  lab notebook `docs/notebook.md`, generate charts/tables.

### 1.2 What stays gated (I prepare fully, then pause at a checkpoint)
- **Push / open PR / merge PR / tag / GitHub release** - tier B. Batched at checkpoints (Sec 5).
- **GitHub repo rename or archive** (the stale-mirror fix) - tier C. Outward-facing + irreversible-ish
  on the user's account; needs explicit per-instance approval, or the user does it.
- **Public posting** (r/LocalLLaMA / HN / forums) - tier C. Explicit approval; I draft, user posts.
- **codex `--dangerously-bypass-approvals-and-sandbox`** - tier C, per-instance AskUserQuestion each time
  (classifier-gated on this box). Not required by any P0/P1 item below.

### 1.3 Guardrails (encoded so I don't have to ask mid-run)
- **R1 - never `git add -A` / `git add .`.** Explicit path staging only. Maintain a hard DROP-list;
  anything on it is never staged: `data/`, `results/**/*ich*`, any `*ich*` or private-clinical script
  (`scripts/train_*_ich*.py`, `scripts/*_rh_*ich*.py`), tokenizer/model weights, `.env`. Before every
  commit: `git status --porcelain` diff review against the DROP-list; if an item is ambiguous, it stays
  unstaged and I note it for the checkpoint. (This is the exact trap from the v1.5 `git add -A` incident.)
- **R2 - no AI tells.** No `Co-Authored-By` trailers; no em-dashes in commit messages, PR bodies, issue
  comments, or release notes.
- **R3 - GB10 hygiene.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on every GPU job. Before a
  GPU run, check free UMA via `torch.cuda.mem_get_info()`; if low, find and kill stray `EngineCore`/vLLM
  by PID (never load a model concurrent with a big download). After a run, release the GPU.
- **R4 - self-verify before "done".** Each item has explicit acceptance criteria. I only mark an item
  done when its criteria pass. If a criterion fails and the fix is out of the item's scope, I stop that
  item, leave it on the branch, and report at the checkpoint rather than pushing something unproven.
- **R5 - checkpoint safety on any train run > 1h.** Periodic adapter checkpoints (every 50 updates,
  rotate last 2).
- **R6 - notebook as I go.** Append observations/measurements/decisions to `docs/notebook.md` per item.

### 1.4 Autonomy tier per workstream (summary; details in Sec 3)
| Item | Tier | Needs GPU | Can run unattended end-to-end? |
|---|---|---|---|
| P0-1 repo hygiene / track patch+plans | A | no | yes, up to the push gate |
| P0-2 stale-mirror repo fix | C (decision) | no | no - needs the canonical-repo decision + a rename |
| P0-3 routed-dequant GPU validation | A | yes | yes (Box A) |
| P1-1 generic-family fallback | A | CPU build + small GPU smoke | yes |
| P1-2 cross-arch serve cells (Nemotron, Mistral) | A | yes | yes (Box B) |
| P1-3 train-time bf16 dequant cache | A | yes | yes |
| P2-1 quality-firming Spider ablations | A | yes | yes |
| P2-2 upstream vLLM patches | B/C | maybe | prep yes; PR-to-vLLM is tier C |
| P2-3 publish push | C | no | draft yes; post no |
| P2-4 hygiene (dedupe is_fp8_module etc.) | A | no | yes |

**Net:** everything except the repo-rename decision (P0-2), the vLLM-upstream PRs (P2-2), and the
public post (P2-3) can run start-to-finish without me stopping - I only pause to batch pushes/PRs at the
checkpoints in Sec 5.

---

## 2. Precondition sweep (first 30 min, tier A, no gate)
Cheap facts that de-risk the rest. Run before touching anything:
1. Confirm both Sparks are idle (`mem_get_info` on A and B; no stray EngineCore). Record free UMA.
2. `pytest -q` on `main` - capture the green baseline (should be ~257 tests) before any change.
3. Read `serve/vllm_patches/nvfp4_emulation_routed_dequant.py` docstring - lift its 4-step GPU
   validation checklist verbatim into the P0-3 acceptance criteria.
4. Read `docs/plans/emulation_speedup_scope.md` + `docs/cross_arch_status.md` for the two pending
   serve cells and the F2 concurrency-table idea.
5. Check current installed vLLM version in the serve venv vs 0.22.1 pin (informs P2-2).

---

## 3. Workstreams

### P0 - do first (days, highest leverage)

#### P0-1  Repo hygiene: track what should be tracked, ignore what shouldn't
- **Tier A. No GPU.** Dependency: none. Gate: push at CP1.
- Steps:
  1. Split the 18 untracked paths into TRACK vs DROP:
     - TRACK: `docs/plans/*.md` (11 planning docs), `serve/vllm_patches/nvfp4_emulation_routed_dequant.py`,
       `results/cross_arch/` (if it contains only non-private eval JSON - verify), and any of the three
       frozen `scripts/train_*` / `scripts/smoke_*` that are **not** ICH/private.
     - DROP (add to `.gitignore`, never stage): `data/`, anything matching R1's DROP-list. The
       `scripts/train_*_ich*.py` / `*_rh_*ich*.py` are private-data trainers -> DROP.
  2. Add/extend `.gitignore` for the DROP set so future `git status` is clean.
  3. Commit the emulation patch **as-is untracked-to-tracked** first (so it can never be `git clean`-ed).
- **Acceptance:** `git status --porcelain` shows only intended files staged; DROP-list greps clean
  (`git diff --cached --name-only | grep -iE 'ich|/data/'` returns nothing); `pytest -q` still green.

#### P0-2  Fix the front door: one canonical GitHub repo
- **Tier C decision + rename. No GPU.** Dependency: none. This is the traction blocker (README points
  at the stale `NvMayMay/nybbloris`; real work is in `NvMayMay/nvfp4-lora-spark`).
- Autonomy note: I **cannot** self-authorize the rename/archive (outward-facing, user's account). What I
  *can* do unattended: prepare both sides of the fix so it's a one-click action once the user picks the
  canonical name.
- Steps (prep, tier A): draft the README/pyproject/CITATION URL edits for **both** possible canonical
  names; draft the archive-notice text for the losing repo. Present at CP1 with a recommendation.
- Gate (tier C): user picks canonical repo; then either the user renames, or approves me running
  `gh repo rename` / archive.
- **Acceptance:** every `github.com/NvMayMay/...` reference in the repo resolves to the live canonical
  repo; the stale repo redirects or carries a pointer.

#### P0-3  GPU-validate the routed-only dequant emulation patch (the serve-speed north-star item)
- **Tier A. GPU (Box A).** Dependency: P0-1 (patch tracked). Gate: land decision at CP2.
- The patch dequantizes only the unique routed experts per forward (defensive fallback to slow path on
  any exception). Target: convert ~0.7 tok/s -> usable single-stream on a 120B MoE.
- Steps (its own docstring checklist): (a) numerical parity vs unpatched emulation on a fixed prompt;
  (b) LoRA-correctness - logprob apply-gate still shows the adapter firing with the patch on;
  (c) speedup timing single-stream + a concurrency sweep (F2: raise `--max-num-seqs`, publish a table);
  (d) re-run the Spider EM harness patched vs unpatched (EM must not regress).
  Use GLM-4.5-Air-NVFP4 (fast iterate) then confirm on the 122B.
- **Acceptance:** parity within tolerance; apply-gate positive; measured single-stream speedup recorded
  with a number; Spider EM delta within noise. Results -> `results/cross_arch/emulation_speedup/` +
  notebook. If any criterion fails, patch stays on-branch, unlanded, reported at CP2 (R4).

### P1 - the "any NVFP4 model" push (2-4 weeks)

#### P1-1  Generic-family fallback instead of hard `SystemExit`
- **Tier A. CPU build + small GPU smoke.** Dependency: P0-1. Gate: CP3.
- Replace the `resolve_family` hard-fail with: synthesize a best-effort family for unknown
  dense/standard-MoE checkpoints (identity key translation, probed peft-scope, probed MoE class via the
  existing inspector + loader heuristics), run it through the existing strict-load + coverage gates, and
  tag the run `UNVERIFIED_FAMILY` in metadata. Add `--family-config family.json` escape hatch.
- **Acceptance:** an intentionally-unregistered but structurally-standard NVFP4 checkpoint trains a few
  steps (coverage gate passes, strict-load passes, metadata tagged); a genuinely-incompatible checkpoint
  still fails fast with a clear message; new CPU tests cover both; existing families unaffected.

#### P1-2  Close the cross-arch serve matrix (Nemotron + Mistral cells)
- **Tier A. GPU (Box B).** Dependency: none (parallel with P0-3 on the other box). Gate: CP3.
- (a) Nemotron non-gated rekey (`up_proj->w1`, no fake gate shard, `down_proj->w2`) + forced-emulation
  logprob apply check. (b) Mistral-Small-4 expert-LoRA serve confirm.
- **Acceptance:** both cells show a positive runtime apply-gate; `docs/cross_arch_status.md` updated to
  green with the evidence; no regression to GLM/Qwen cells.

#### P1-3  Train-time bf16 dequant cache for models with headroom
- **Tier A. GPU.** Dependency: none. Gate: CP3.
- Extend the eval-path capped bf16 cache (`linear.py:163-201`) to training under an opt-in memory cap
  (`--train-dequant-cache-gb`, LRU per-module). Small/mid models (<=32B) keep bf16 weights resident ->
  near-bf16 step time; 120B stays on the on-the-fly path.
- **Acceptance:** measured train-step speedup on an 8-32B NVFP4 model at a fixed cache cap, loss curve
  unchanged vs uncached within noise, 120B path untouched when the flag is off (default off).

### P2 - credibility + distribution (parallel, mixed effort)

#### P2-1  Firm the ~78% quality claim on public data
- **Tier A. GPU.** Dependency: none. Gate: CP3/CP4.
- Re-run the NVFP4-vs-bf16-LoRA ablation on the public Spider harness for 2 bases where bf16 fits on-box
  (e.g. Llama-8B, Mistral-24B). Then cheap gap-closers: make `--mask-prompt-labels` default, rank/alpha
  sweep; one LoftQ-style quant-error-init experiment on a <=32B base (needs bf16 base, feasible there).
- **Acceptance:** committed Spider eval JSON for both bases; a recovery-% number backed by public data
  (not just the single private ICH run); any default change gated behind a passing ablation.

#### P2-2  Upstream the vLLM patches (un-pin from 0.22.1)
- **Tier B/C.** Dependency: P0-3 (routed dequant proven). Prep is tier A; a PR to vLLM is tier C.
- Candidates: routed-only emulation dequant, marlin-repack memory fix, non-gated-MoE LoRA w13 handling.
  First (tier A, 0.5 day): check whether vLLM >=0.23 changed the NVFP4-MoE-backend LoRA story before
  investing. Draft PRs locally; opening them upstream is tier C.
- **Acceptance:** a written up/down call on each candidate + at least one PR-ready branch; version
  tripwire so a vLLM bump that breaks the patch is caught in CI.

#### P2-3  Publish push (traction gate)
- **Tier C.** Dependency: P0-2 (canonical repo) + P0-3 (a real speed number helps). Draft tier A.
- Draft the r/LocalLLaMA / HN post around the Spider before/after + the "your 4-bit fine-tune silently
  vanished and the server never told you" story. I draft; user posts.
- **Acceptance:** a ready-to-post draft + the linked repro landing on the canonical repo.

#### P2-4  Hygiene
- **Tier A. No GPU.** Dependency: none. Gate: CP1/CP3.
- Dedupe `is_fp8_module` (currently in both `adapter_keys.py:158` and `loader.py:93` - drift silently
  desyncs preflight from load): make one import the other, add a cross-check test. Collapse duplicated
  registry entries (`qwen3_5_moe`/`_text`, `mistral3`/`4`) into shared dicts. Add a `scripts/`-exists
  guard in `cli.py`. Stand up a self-hosted nightly GPU smoke on the Spark (the pending "GPU ladder").
- **Acceptance:** single source of truth for `is_fp8_module` with a test; registry dedup with families
  unchanged; GPU-smoke script runnable on the box.

### P3 - parked (only on external demand signal)
Fused dequant-GEMM Triton training kernel (the real 120B train-throughput fix; v2 headline) · Path 1
descriptor-cliff spike (1 wk; the ~2-4k seq-len ceiling on 120B is a real SFT limit) · **keep Path 2
static-CUDA-graph engine parked** (14-22 wk, oversized pre-traction) · TE NVFP4 kernels dead on sm_121.

---

## 4. Execution sequence + box allocation
Two Sparks -> two GPU tracks run in parallel; CPU items fill gaps.

```
Day 0 (CPU, unattended):  precondition sweep -> P0-1 hygiene -> P2-4 dedupe   -> [CP1: push gate]
Week 1:
  Box A (GPU):  P0-3 routed-dequant validation (GLM-Air -> 122B)              -> [CP2: land gate]
  Box B (GPU):  P1-2 Nemotron + Mistral serve cells
  CPU (fill):   P0-2 prep + P1-1 generic-family build
Week 2-3:
  Box A/B (GPU): P1-3 bf16 cache bench + P2-1 Spider ablations (split across boxes)
  CPU:           P2-2 vLLM-version recon + patch drafts                       -> [CP3: PR batch]
Week 3-4:
  P2-3 publish draft + P2-2 upstream prep                                     -> [CP4: release/post]
```
Dependencies that force order: P0-1 before anything committed; P0-3 before P2-2 (dequant) and P2-3
(speed number); P0-2 before P2-3 (canonical repo).

---

## 5. Human-gate checkpoints (where I pause and batch)
- **CP1 (after Day 0):** review the selective-commit set (P0-1 + P2-4) before push; pick the canonical
  repo (P0-2). One approval -> I push the branch / open the hygiene PR.
- **CP2 (after P0-3):** review the routed-dequant numbers; decide land vs iterate. Approval -> merge.
- **CP3 (after P1 + P2-1):** review generic-family + cross-arch + ablation results; batch the PRs.
- **CP4 (release/publish):** cut v1.7 (version bump + tag + release notes) and approve the public post.

Between checkpoints I run unattended under G1-G3 and R1-R6. If I hit an out-of-scope failure I stop that
item and surface it at the next checkpoint rather than pushing something unproven.

---

## 6. Risks / rollback
- **DROP-list leak** (private ICH/data into a public commit) - mitigated by R1 explicit staging +
  `.gitignore` + pre-commit grep; rollback is `git reset --soft` + `restore --staged` (the v1.5 drill).
- **Untracked patch loss** - P0-1 tracks it first thing.
- **Emulation patch regresses correctness** - apply-gate + Spider EM are hard gates (R4); patch stays
  unlanded on failure.
- **vLLM pin drift** - P2-2 version tripwire; recon before invest.
- **Traction never fires** - P0-2 + P2-3 directly target it; it is the roadmap's own gate for P3.
