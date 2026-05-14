#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_launch_utils.sh
source "${SCRIPT_DIR}/_launch_utils.sh"

REPO_ROOT="$(resolve_repo_root)"
CONFIG_FILE="${1:?Usage: bash scripts/launch_train.sh <config.yaml>}"
CONFIG_FILE="$(resolve_config_path "${REPO_ROOT}" "${CONFIG_FILE}")"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29601}"
AUTO_GPU_MAX_USED_MB="${AUTO_GPU_MAX_USED_MB:-1024}"
prepare_cuda_visible_devices "${NPROC}" "${AUTO_GPU_MAX_USED_MB}"

echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-default} (nproc=${NPROC})"

cd "${REPO_ROOT}"
torchrun \
  --standalone \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  "${REPO_ROOT}/scripts/train_scar.py" \
  --config "${CONFIG_FILE}"
