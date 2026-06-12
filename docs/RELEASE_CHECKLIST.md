# Release checklist

What to run before tagging a release or merging a substantial branch. Ordered
cheapest-first; everything above the GPU ladder runs on any machine.

## CPU gate (CI runs these on every push)

```bash
python -m compileall -q nvfp4_lora scripts train serve smoke_tests tests plots
bash -n serve/*.sh serve/diagnostics/*.sh scripts/*.sh
python -m pytest tests/ -q
```

## Static checks

- `grep -rn "$HOME\|/home/" scripts/*.sh serve/*.sh` finds machine-local paths
  that leaked outside `serve/local_env.sh` (which is gitignored).
- `git status --short` is empty of surprises: every new file is either
  committed deliberately or documented as a local artifact.
- README quickstart commands reference scripts and flags that actually exist.

## GPU validation ladder (needs a Spark + a real NVFP4 checkpoint)

Run in order; each step is cheaper than the next and catches a different
failure class:

1. **Inspect** - `scripts/inspect_nvfp4_checkpoint.py <model> --target-modules ... --deep`
   exits 0 and the verdict matches expectations.
2. **Trainer dry-run** - `scripts/train_nvfp4_lora.py ... --dry-run` completes
   one synthetic forward+backward at the production (batch, max_length) and
   logs the memory peak.
3. **3-step train** - `--max-train-examples 8 --max-steps 3 --eval-every 0
   --checkpoint-every 0` to a throwaway output dir; confirm
   `target_coverage.json` and a loadable adapter appear.
4. **Save/resume** - run with `--checkpoint-every 1 --max-steps 2`, kill,
   resume with `--resume-from <output>/checkpoint_step_1`; losses continue,
   no `resume_fastforward_done` anomalies.
5. **Merge dry-run** - the matching merge script with `--dry-run` against the
   produced adapter: 100% coverage, expected scale groups.
6. **Merge + validate** - full merge, then `scripts/validate_merge.py`;
   worst merge_cosine in line with previous releases (>= 0.999 typical).
7. **Serve smoke** - the relevant `serve/` launcher comes up and answers one
   `/v1/chat/completions` request.

## Documentation

- README throughput/memory tables still match the shipped results artifacts.
- `docs/SUPPORTED_TOPOLOGIES.md` reflects any newly supported or newly
  rejected layout.
- New failure signatures observed during the release work are added to
  `docs/TROUBLESHOOTING.md`.
