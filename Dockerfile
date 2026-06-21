# Frozen nybbloris install on the NGC vLLM image (the measured-good serve runtime
# for compressed-tensors NVFP4 on GB10; see docs measured_evidence.md section 7).
#
# Build:
#   docker build -t nybbloris:dev .
# Inspect (pure pre-flight, no GPU needed):
#   docker run --rm -v /path/to/models:/models nybbloris:dev \
#     inspect --base-model-dir /models/<base> --adapter-dir /models/<adapter>
# Serve (dynamic LoRA, needs GPU):
#   docker run --gpus all --ipc=host --network host \
#     -v /path/to/models:/models -v $PWD:/repo nybbloris:dev \
#     serve --base-model-dir /models/<base> --adapter-dir /models/<adapter> \
#     --launcher /repo/serve/run_qwen35_122b_nvfp4_dynamic_lora_docker.sh
FROM nvcr.io/nvidia/vllm:26.05-py3

WORKDIR /opt/nybbloris
COPY . /opt/nybbloris

# torch / vllm / transformers / safetensors are already in the base image; install
# the package itself without re-resolving them. `nybbloris inspect` is pure stdlib.
RUN pip install --no-deps -e .

ENTRYPOINT ["nybbloris"]
CMD ["--help"]
