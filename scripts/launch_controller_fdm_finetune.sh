#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_launch_utils.sh
source "${SCRIPT_DIR}/_launch_utils.sh"

REPO_ROOT="$(resolve_repo_root)"
CONFIG_FILE="${1:?Usage: bash scripts/launch_controller_fdm_finetune.sh <config.yaml> <controller_ckpt.pt> [output_dir]}"
CONTROLLER_CKPT="${2:?Usage: bash scripts/launch_controller_fdm_finetune.sh <config.yaml> <controller_ckpt.pt> [output_dir]}"
OUTPUT_DIR="${3:-}"
CONFIG_FILE="$(resolve_config_path "${REPO_ROOT}" "${CONFIG_FILE}")"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CONTROLLER_CKPT}" ]]; then
  echo "Error: controller checkpoint not found: ${CONTROLLER_CKPT}" >&2
  exit 1
fi

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29631}"
AUTO_GPU_MAX_USED_MB="${AUTO_GPU_MAX_USED_MB:-1024}"
prepare_cuda_visible_devices "${NPROC}" "${AUTO_GPU_MAX_USED_MB}"

cmd=(
  torchrun
  --standalone
  --nproc_per_node "${NPROC}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/scripts/finetune_action_controller_fdm.py"
  --config "${CONFIG_FILE}"
  --controller-ckpt "${CONTROLLER_CKPT}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

cd "${REPO_ROOT}"
"${cmd[@]}"
