#!/usr/bin/env bash

resolve_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

resolve_config_path() {
  local repo_root="$1"
  local config_file="$2"

  if [[ "${config_file}" = /* ]]; then
    printf '%s\n' "${config_file}"
  elif [[ -f "${repo_root}/${config_file}" ]]; then
    printf '%s\n' "${repo_root}/${config_file}"
  elif [[ -f "${repo_root}/configs/${config_file}" ]]; then
    printf '%s\n' "${repo_root}/configs/${config_file}"
  else
    printf '%s\n' "${config_file}"
  fi
}

pick_idle_gpus() {
  local required_gpus="$1"
  local max_used_mb="${2:-1024}"
  local -a candidates=()

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  while IFS=',' read -r gpu_index used_mb free_mb; do
    gpu_index="${gpu_index//[[:space:]]/}"
    used_mb="${used_mb//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    [[ -z "${gpu_index}" ]] && continue
    if (( used_mb <= max_used_mb )); then
      candidates+=("${free_mb}:${gpu_index}")
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
  )

  if (( ${#candidates[@]} < required_gpus )); then
    return 2
  fi

  mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | sort -t: -k1,1nr)

  local -a selected=()
  for candidate in "${candidates[@]}"; do
    selected+=("${candidate#*:}")
    if (( ${#selected[@]} == required_gpus )); then
      break
    fi
  done

  (IFS=,; printf '%s' "${selected[*]}")
}

prepare_cuda_visible_devices() {
  local nproc="$1"
  local max_used_mb="${2:-1024}"

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(pick_idle_gpus "${nproc}" "${max_used_mb}")" || {
      echo "Error: could not find ${nproc} idle GPU(s)." >&2
      return 1
    }
    export CUDA_VISIBLE_DEVICES
    echo "Auto-selected GPUs: ${CUDA_VISIBLE_DEVICES}"
  fi
}
