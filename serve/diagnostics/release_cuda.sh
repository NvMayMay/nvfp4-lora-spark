#!/usr/bin/env bash
# Try to recover CUDA-visible memory after a diagnostic run on Spark UMA.
#
# On the integrated GB10 GPU on DGX Spark, the nvidia kernel module's
# internal allocator can hold ~40 GB of state after a process crashes,
# even though no userspace process is holding the device. This script
# attempts (in increasing order of invasiveness):
#
#   1. Clean dead IPC sockets in /tmp from prior vLLM runs.
#   2. Drop OS page cache (releases mmap'd safetensors).
#   3. Restart nvidia-persistenced (rarely helps but no risk).
#   4. Report whether memory recovered.
#
# If after this cuda_free is still well below the boot-baseline (~122 GB),
# only a reboot will fully recover. This script intentionally does NOT
# rmmod nvidia/nvidia_uvm/nvidia_drm/nvidia_modeset because on Spark the
# display server holds refs to those modules and unloading them would
# kill the desktop session.
#
# Usage:
#   sudo ./release_cuda.sh

set -u

if [ "$(id -u)" -ne 0 ]; then
  echo "[release_cuda] needs sudo: re-run with: sudo $0"
  exit 1
fi

VENV_PY=/path/to/venvs/serve/bin/python

measure() {
  local tag="$1"
  "$VENV_PY" -c "
import torch, psutil
f, t = torch.cuda.mem_get_info()
v = psutil.virtual_memory()
print(f'[$tag] cuda_free={f/1e9:.2f}GB ram_avail={v.available/1e9:.2f}GB')
"
}

measure before

echo '[step 1] killing any leftover vllm/diag/cicc/nvcc processes...'
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
pkill -9 -f 'diag_vllm_safe'   2>/dev/null
pkill -9 -f 'diag_alloc_micro' 2>/dev/null
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null
pkill -9 -f 'nvcc.*flashinfer' 2>/dev/null
pkill -9 -f '^cicc'            2>/dev/null
pkill -9 -f 'tail -F.*diag_'   2>/dev/null
sleep 2

echo '[step 2] cleaning dead /tmp ZMQ IPC sockets...'
find /tmp -maxdepth 1 -name '[a-f0-9]*-[a-f0-9]*-*-*' -type s -mmin +1 -delete 2>/dev/null || true

echo '[step 3] dropping OS page cache (sync + drop_caches=3)...'
sync
echo 3 > /proc/sys/vm/drop_caches

echo '[step 4] restart nvidia-persistenced (no-op for memory but resets daemon state)...'
systemctl restart nvidia-persistenced 2>/dev/null || true
sleep 1

measure after

echo ''
echo 'If cuda_free is still well below ~120 GB, only a reboot will fully recover.'
echo 'Recommendation: reboot if you need a clean baseline for the next experiment.'
