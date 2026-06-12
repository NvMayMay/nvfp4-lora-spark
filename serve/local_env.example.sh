# Machine-local roots for the serve launchers. Copy to serve/local_env.sh
# (gitignored) and point at your own layout, or export the variables yourself.
#
#   NVFP4_MODELS_DIR    directory holding the downloaded model checkpoints
#   NVFP4_ADAPTERS_DIR  directory holding trained LoRA adapter outputs
#   NVFP4_SERVE_VENV    the vLLM serving virtualenv (see README Quickstart)

export NVFP4_MODELS_DIR="$HOME/models"
export NVFP4_ADAPTERS_DIR="$HOME/adapters"
export NVFP4_SERVE_VENV="$HOME/nvfp4-lora-spark/.venv-serve"
