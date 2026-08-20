#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

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

CUSTOM_NODES_DIR="${COMFYUI_ROOT}/custom_nodes"
mkdir -p "${CUSTOM_NODES_DIR}"

find_python() {
    if [[ -n "${COMFYUI_PYTHON:-}" && -x "${COMFYUI_PYTHON}" ]]; then
        printf '%s\n' "${COMFYUI_PYTHON}"
        return 0
    fi

    local candidate
    local candidates=("${COMFYUI_ROOT}/venv/bin/python" "${COMFYUI_ROOT}/.venv/bin/python" /workspace/venv/bin/python /opt/venv/bin/python)
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    command -v python3 || command -v python
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python was not found. Set COMFYUI_PYTHON=/path/to/python." >&2
    exit 1
fi

install_repo() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local destination="${CUSTOM_NODES_DIR}/${name}"

    if [[ -d "${destination}/.git" ]]; then
        if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
            echo "WARN: ${name} has local changes; leaving its checkout untouched."
        else
            git -C "${destination}" fetch --depth 1 origin "${ref}" || return 1
            git -C "${destination}" checkout --detach FETCH_HEAD || return 1
        fi
    elif [[ -e "${destination}" ]]; then
        echo "ERROR: ${destination} exists but is not a Git checkout." >&2
        return 1
    else
        git clone "${url}" "${destination}" || return 1
        git -C "${destination}" fetch --depth 1 origin "${ref}" || return 1
        git -C "${destination}" checkout --detach FETCH_HEAD || return 1
    fi

    if [[ -f "${destination}/requirements.txt" ]]; then
        "${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${destination}/requirements.txt" || return 1
    fi
}

MINIMAX_REFPACK_REF="${MINIMAX_REFPACK_REF:-7012734eabf6f98063d6eaf8ce1f9264ee803664}"
KJNODES_REF="${KJNODES_REF:-3f20054214fec9f9234fd3841ae6f1e4287948f6}"
RGTHREE_REF="${RGTHREE_REF:-f4bf78648bf7f72fb6ff1365a431ed510931b21a}"
VHS_REF="${VHS_REF:-4ee72c065db22c9d96c2427954dc69e7b908444b}"

status=0
install_repo "ComfyUI-MiniMaxRefPack" "https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack.git" "${MINIMAX_REFPACK_REF}" || status=1
install_repo "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git" "${KJNODES_REF}" || status=1
install_repo "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git" "${RGTHREE_REF}" || status=1
install_repo "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" "${VHS_REF}" || status=1

if [[ -f "${PROJECT_DIR}/requirements.txt" ]]; then
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${PROJECT_DIR}/requirements.txt" || status=1
fi

WORKFLOW_DEST="${COMFYUI_ROOT}/user/default/workflows/HmAI-Nxdify-Nodes"
mkdir -p "${WORKFLOW_DEST}"
for workflow in "${PROJECT_DIR}"/workflows/*.json; do
    [[ -f "${workflow}" ]] || continue
    cp -f "${workflow}" "${WORKFLOW_DEST}/${workflow##*/}" || status=1
done

check_path() {
    local label="$1"
    local path="$2"
    if [[ -e "${path}" ]]; then
        printf '%-30s OK\n' "${label}:"
    else
        printf '%-30s MISSING\n' "${label}:"
        status=1
    fi
}

echo
echo "MiniMax Auto Prompt verification"
check_path "MiniMaxRefPack" "${CUSTOM_NODES_DIR}/ComfyUI-MiniMaxRefPack/minimax_refpack/nodes.py"
check_path "Two-stage prompt node" "${PROJECT_DIR}/minimax_autoprompt_nodes.py"
check_path "KJNodes" "${CUSTOM_NODES_DIR}/ComfyUI-KJNodes"
check_path "rgthree" "${CUSTOM_NODES_DIR}/rgthree-comfy"
check_path "VideoHelperSuite" "${CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite"
check_path "I2V workflow/module" "${WORKFLOW_DEST}/I2V_AutoPrompt_Module.json"
check_path "R2V workflow/module" "${WORKFLOW_DEST}/R2V_AutoPrompt_Module.json"

if [[ "${status}" -eq 0 ]]; then
    printf '%-30s OK\n' "Python requirements:"
    echo "Restart ComfyUI to load the new nodes and workflows."
else
    printf '%-30s ERROR\n' "Python requirements:"
fi

exit "${status}"
