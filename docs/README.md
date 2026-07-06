# Documentation

The guide index for **nybbloris** — LoRA fine-tuning and runtime-LoRA serving for NVFP4 MoE
on a single DGX Spark (GB10). Start at the [project README](../README.md) for the pitch and
the quickstart; this page maps everything else.

## Get started

| Doc | What it covers |
|---|---|
| [REPRODUCE_SPIDER.md](../REPRODUCE_SPIDER.md) | Reproduce a real before/after in ~30 min on a public 8B + Spider text-to-SQL. The fastest end-to-end proof. |
| [REPRODUCE.md](../REPRODUCE.md) | The exact stack: dependencies, versions, CUDA, and the licensing breakdown. |
| [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md) | Full CLI walkthrough: `train → inspect → serve → verify`. |

## Serve

| Doc | What it covers |
|---|---|
| [SERVING.md](SERVING.md) | The blessed host-venv serve recipe, UMA gotchas, and the runtime-by-checkpoint table (which base+adapter serves live vs. merge-only). |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failure-signature playbook: match the error, apply the fix. |

## Models and topologies

| Doc | What it covers |
|---|---|
| [SUPPORTED_TOPOLOGIES.md](SUPPORTED_TOPOLOGIES.md) | The exact NVFP4 checkpoint-layout contract a family must satisfy. |
| [PORTING.md](PORTING.md) | How to add another NVFP4 family (one registry entry + the inspect-first workflow). |
| [CONTEXT_FIT_MATRIX.md](CONTEXT_FIT_MATRIX.md) | Which model fits at which context length under the unified trainer. |
| [GLM45_AIR_FINETUNING.md](GLM45_AIR_FINETUNING.md) | End-to-end fine-tune + serve runbook for GLM-4.5-Air (106B-A12B). |

## Measurements

| Doc | What it covers |
|---|---|
| [BENCHMARKS.md](BENCHMARKS.md) | Every measured table: training memory, long-context fits, throughput, concurrency (with committed eval JSON). |
| [LONG_CONTEXT_EXPERIMENTS.md](LONG_CONTEXT_EXPERIMENTS.md) | Long-context training-fit experiments on the 120B-class Super model. |

## Project

| Doc | What it covers |
|---|---|
| [../CHANGELOG.md](../CHANGELOG.md) | Release history (v1.0 onward). |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, running the tests, and how to propose a change. |
| [LICENSES.md](LICENSES.md) | Licenses, provenance, and the artifact-redistribution policy. |
| [LESSONS.md](LESSONS.md) | Engineering lessons and deviations recorded during the build. |
| [PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md) | The plan for closing the NVFP4-to-BF16 throughput gap. |
| [PHASE2.md](PHASE2.md) | Future work: dynamic LoRA at CUTLASS speeds. |
| [cross_arch_status.md](cross_arch_status.md) | Cross-architecture expert-LoRA status board. |

> `docs/plans/` holds internal design drafts and scoping notes; they are working documents,
> not user-facing guides.
