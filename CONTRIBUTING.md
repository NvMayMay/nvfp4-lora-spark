# Contributing

Thanks for your interest in nybbloris. Issues and pull requests are welcome.

For larger changes — a new model-family loader, a native FP4 training path, dynamic-LoRA
work — please **open an issue first** to align on scope before writing code. Small fixes
(docs, a failing edge case, a clear bug) can go straight to a PR.

## Dev setup

```bash
git clone https://github.com/NvMayMay/nvfp4-lora-spark
cd nvfp4-lora-spark
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"          # library + the CPU test extras (pytest, pillow)
nybbloris doctor                  # environment pre-flight
```

The CLI's `train`/`serve` subcommands shell out to repo-relative `scripts/`, so develop
against the source tree (an editable install), not a plain wheel. The GPU training/serving
stack is installed separately per [REPRODUCE.md](REPRODUCE.md).

## Tests

The suite under [`tests/`](tests/) is **CPU-only by construction** — it never touches CUDA,
so it runs anywhere and is what CI gates on:

```bash
python -m pytest tests/ -q
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), "CPU tests") runs on every push to
`main` and every PR. It does three things, all of which should pass locally before you open a
PR:

1. `python -m compileall` over the package + scripts (syntax).
2. `bash -n` over the shell launchers in `serve/` and `scripts/`.
3. `python -m pytest tests/`.

The GPU smoke tests in [`smoke_tests/`](smoke_tests/) require a real GB10 and an NVFP4
checkpoint; they are intentionally **not** in CI. Run them by hand when a change touches the
dequant kernel, the loader, or `NVFP4LoRALinear`.

## Pull requests

- Keep the CPU suite green, and add a test when you change behaviour behind the binding
  contract (`nvfp4_lora/adapter_keys.py`, `nybbloris/plan.py`, the `inspect` verdicts).
- Update the relevant doc when you change a user-facing surface — the CLI, a serve recipe, or
  the supported-model table. The [docs index](docs/README.md) maps where things live.
- Describe *what you validated* and *on what hardware* in the PR body. A capability claim in
  this repo is expected to come with evidence (an eval JSON, a logprob delta, a fit
  measurement), not just "works on my box."

## Adding a new NVFP4 family

Most of the work is one registry entry. Follow [docs/PORTING.md](docs/PORTING.md): run
`scripts/inspect_nvfp4_checkpoint.py` on the checkpoint first, add the family to
`nvfp4_lora/families.py`, and let the strict-load + target-coverage gates catch a layout
mismatch.

## License

By contributing, you agree that your contributions are licensed under the
[Apache 2.0](LICENSE) license that covers this project.
