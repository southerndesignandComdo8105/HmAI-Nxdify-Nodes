#!/usr/bin/env bash
set -uo pipefail

find_comfyui() {
    if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
        printf '%s\n' "${COMFYUI_DIR}"
        return 0
    fi

    local candidate
    for candidate in /ComfyUI /workspace/ComfyUI "${PWD}" "${PWD}/ComfyUI"; do
        if [[ -f "${candidate}/main.py" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

COMFYUI_ROOT="$(find_comfyui || true)"
if [[ -z "${COMFYUI_ROOT}" ]]; then
    echo "ERROR: ComfyUI was not found. Set COMFYUI_DIR=/path/to/ComfyUI." >&2
    exit 1
fi

MODELS_DIR="${COMFYUI_ROOT}/models"
downloaded=0
skipped=0
failed=0

download_file() {
    local relative_path="$1"
    local url="$2"
    local destination="${MODELS_DIR}/${relative_path}"
    local partial="${destination}.part"

    mkdir -p "$(dirname "${destination}")"

    if [[ -s "${destination}" ]]; then
        echo "SKIP       ${relative_path}"
        skipped=$((skipped + 1))
        return 0
    fi

    echo "DOWNLOAD   ${relative_path}"
    if command -v curl >/dev/null 2>&1; then
        if ! curl --fail --location --retry 3 --continue-at - --output "${partial}" "${url}"; then
            echo "ERROR      ${relative_path}" >&2
            failed=$((failed + 1))
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget --continue --output-document "${partial}" "${url}"; then
            echo "ERROR      ${relative_path}" >&2
            failed=$((failed + 1))
            return 1
        fi
    else
        echo "ERROR: curl or wget is required." >&2
        failed=$((failed + 1))
        return 1
    fi

    mv -f "${partial}" "${destination}"
    downloaded=$((downloaded + 1))
}

# Shared MiniMax H3 assets used by both integrated workflows.
download_file "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
download_file "vae/minimax_h3_video_vae_fp16.safetensors" "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors"
download_file "vae/minimax_h3_audio_vae_fp32.safetensors" "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors"

# I2V / FL2VA assets.
download_file "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
download_file "loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors" "https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"

# R2V / Ref2VA assets.
download_file "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors" "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
download_file "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# Optional preview VAE selected by both integrated workflows.
download_file "vae_approx/taeh3.safetensors" "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors"

echo
echo "Downloaded: ${downloaded}"
echo "Skipped:    ${skipped}"
echo "Failed:     ${failed}"

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
