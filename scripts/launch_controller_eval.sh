#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_launch_utils.sh
source "${SCRIPT_DIR}/_launch_utils.sh"

REPO_ROOT="$(resolve_repo_root)"
CONTROLLER_CKPT="${1:?Usage: bash scripts/launch_controller_eval.sh <controller_ckpt.pt> [output_dir]}"
OUTPUT_DIR="${2:-}"

if [[ ! -f "${CONTROLLER_CKPT}" ]]; then
  echo "Error: controller checkpoint not found: ${CONTROLLER_CKPT}" >&2
  exit 1
fi

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"
AUTO_GPU_MAX_USED_MB="${AUTO_GPU_MAX_USED_MB:-1024}"
prepare_cuda_visible_devices "${NPROC}" "${AUTO_GPU_MAX_USED_MB}"

cmd=(
  torchrun
  --standalone
  --nproc_per_node "${NPROC}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/scripts/eval_action_controller.py"
  --controller-ckpt "${CONTROLLER_CKPT}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

cd "${REPO_ROOT}"
"${cmd[@]}"
