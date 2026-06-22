# Frozen nybbloris image: inspect / serve (dynamic LoRA, + runtime verify) / train.
#
# The base image IS the vLLM serve runtime, so choose it to match your checkpoint
# (MEASURED -- docs/measured_evidence.md sections 7a, 7h):
#   * compressed-tensors NVFP4 (e.g. the 122B): vLLM >= 0.19 (NGC vllm:26.04+). DEFAULT below.
#   * ModelOpt NVFP4 with a quantized MoE / lm_head (the canonical Qwen3.6-35B-A3B-NVFP4):
#       needs vLLM >= 0.22.1. NGC tops out at 0.20.1 and an aarch64 0.22.1 build is
#       intentionally out of scope here -- serve that base from a 0.22.1 host venv
#       (serve/README.md), or build this image on your own 0.22.1 base:
#         docker build --build-arg VLLM_BASE=<image-with-vllm-0.22.1> -t nybbloris:serve .
#
# Build (default NGC base):
#   docker build -t nybbloris:dev .
#
# Inspect (pure pre-flight, no GPU -- the binding contract):
#   docker run --rm -v /models:/models nybbloris:dev \
#     inspect --base-model-dir /models/<base> --adapter-dir /models/<adapter>
#
# Serve dynamic LoRA, auto-re-keying a silent no-op, with an optional runtime verify
# (needs GPU; --network host exposes the port; mount adapters read-write so --rekey
# auto can write the re-keyed copy):
#   docker run --gpus all --ipc=host --network host \
#     -v /models:/models nybbloris:dev \
#     serve --base-model-dir /models/<base> \
#           --adapter ich=/models/<adapter> --rekey auto --port 8001 \
#           [--verify --val-file /models/<val>.jsonl]
ARG VLLM_BASE=nvcr.io/nvidia/vllm:26.05-py3
FROM ${VLLM_BASE}

# flashinfer JIT-compiles the FP8 / GDN kernels at first serve and needs ninja + nvcc
# on PATH. nvcc ships in the NGC CUDA base; ninja is the missing piece. Baking it in
# is harmless when unused (e.g. on the 0.20.1 base) and unblocks a 0.22.1 base.
RUN (command -v ninja >/dev/null 2>&1) \
    || (apt-get update \
        && apt-get install -y --no-install-recommends ninja-build \
        && rm -rf /var/lib/apt/lists/*) \
    || true
ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

WORKDIR /opt/nybbloris
COPY . /opt/nybbloris

# torch / vllm / transformers / safetensors are already in the base image; install
# the package itself without re-resolving them. `nybbloris inspect` is pure stdlib.
# The in-container `vllm` is on PATH, so `nybbloris serve` uses it with no extra flags.
RUN pip install --no-deps -e .

ENTRYPOINT ["nybbloris"]
CMD ["--help"]
