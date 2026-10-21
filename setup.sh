#!/usr/bin/env bash
# gimpsegany setup: backend venv, Python deps, checkpoints, plugin install.
#
#   ./setup.sh [--venv DIR] [--with-models] [--models-dir DIR]
#              [--install-plugin] [--plugindir DIR] [--dry-run]
#
# Defaults: project-local .venv, no model downloads, no plugin install.
# Env knobs: TORCH_INDEX_URL (e.g. the ROCm wheel index), PIP_EXTRA_ARGS.
#
#   ./setup.sh --with-models --install-plugin
#
# Verifies with: ./test_sam3.sh, ./test_sam12.sh, <venv>/bin/python -m pytest tests/
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv"
MODELS_DIR="$HOME/gimpsegany-models"
WITH_MODELS=0
INSTALL_PLUGIN=0
PLUGIN_DIR=""
DRY=0

log() { printf '%s\n' "$*"; }
fail() { log "ERROR: $*"; exit 1; }
run() {
    if [ "$DRY" -eq 1 ]; then
        log "  [dry-run] $*"
    else
        log "  \$ $*"
        "$@"
    fi
}

usage() {
    sed -n '2,12p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --venv) VENV="$2"; shift 2 ;;
        --with-models) WITH_MODELS=1; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --install-plugin) INSTALL_PLUGIN=1; shift ;;
        --plugindir) PLUGIN_DIR="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage 0 ;;
        *) fail "unknown option: $1 (see --help)" ;;
    esac
done

command -v python3 >/dev/null || fail "python3 not found (need 3.10+)"
command -v git >/dev/null || fail "git not found (needed for the SAM1/SAM2 backends)"
command -v curl >/dev/null || fail "curl not found (needed for the SAM1 checkpoint)"

log "venv: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    run python3 -m venv "$VENV"
else
    log "reusing existing venv"
fi
PY="$VENV/bin/python"
run "$PY" -m pip install --upgrade pip

TORCH_ARGS=()
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    TORCH_ARGS=(--index-url "$TORCH_INDEX_URL")
fi
# shellcheck disable=SC2128
run "$PY" -m pip install ${PIP_EXTRA_ARGS:-} "${TORCH_ARGS[@]}" \
    torch torchvision \
    "transformers>=5.10" safetensors pillow opencv-python-headless \
    huggingface_hub pytest
run "$PY" -m pip install ${PIP_EXTRA_ARGS:-} \
    "git+https://github.com/facebookresearch/segment-anything.git" \
    "git+https://github.com/facebookresearch/sam2.git"

if [ "$WITH_MODELS" -eq 1 ]; then
    log "models: $MODELS_DIR"
    run mkdir -p "$MODELS_DIR/sam3.1" "$MODELS_DIR/sam2.1-hiera-large"
    # SAM 3.1: ungated mirror, BF16 weights (no ROCm kernels needed).
    run "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/sam3.1', local_dir='$MODELS_DIR/sam3.1', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
    # SAM 2.1 large: only the Meta-layout .pt is loadable by the sam2 package
    # (the folder's model.safetensors is transformers-native).
    run "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-large', local_dir='$MODELS_DIR/sam2.1-hiera-large', allow_patterns=['*.pt', '*.yaml'])"
    # SAM1 vit_h.
    if [ ! -f "$MODELS_DIR/sam_vit_h_4b8939.pth" ]; then
        run curl -sfL -o "$MODELS_DIR/sam_vit_h_4b8939.pth" \
            "https://dl.fbaipublicfiles.com/segment-anything/sam_vit_h_4b8939.pth"
    else
        log "reusing sam_vit_h_4b8939.pth"
    fi
fi

if [ "$INSTALL_PLUGIN" -eq 1 ]; then
    if [ -z "$PLUGIN_DIR" ]; then
        case "$(uname -s)" in
            Linux) GIMP_BASE="$HOME/.config/GIMP" ;;
            Darwin) GIMP_BASE="$HOME/Library/Application Support/GIMP" ;;
            *) fail "unsupported OS for plugin autodetect (pass --plugindir)" ;;
        esac
        LATEST="$(ls -d "$GIMP_BASE"/3* 2>/dev/null | sort -V | tail -1 || true)"
        [ -n "$LATEST" ] || fail "no GIMP 3 user dir under $GIMP_BASE (pass --plugindir)"
        PLUGIN_DIR="$LATEST/plug-ins/seganyplugin"
    fi
    log "plugin: $PLUGIN_DIR"
    run mkdir -p "$PLUGIN_DIR"
    run cp "$REPO/seganyplugin.py" "$REPO/seganybridge.py" "$PLUGIN_DIR/"
    run chmod +x "$PLUGIN_DIR/seganyplugin.py"
    SETTINGS="$PLUGIN_DIR/segany_settings.json"
    if [ ! -f "$SETTINGS" ]; then
        log "writing default settings (Python3 Path -> new venv)"
        if [ "$DRY" -eq 1 ]; then
            log "  [dry-run] write $SETTINGS"
        else
            cat >"$SETTINGS" <<EOF
{"pythonPath": "$PY", "modelType": "auto", "checkPtPath": null, "maskType": "Multiple", "segType": "Auto", "isRandomColor": false, "maskColor": [255, 0, 0, 255], "selPtCnt": 10, "segRes": "Medium", "cropNLayers": 0, "minMaskArea": 0, "textPrompt": ""}
EOF
        fi
    else
        log "keeping existing segany_settings.json (set Python3 Path to $PY)"
    fi
    log "restart GIMP, then Image > Segment Anything Layers"
fi

log "done. verify with: ./test_sam3.sh <img> $PY <models>/sam3.1"
log "                    ./test_sam12.sh [img] [$PY]"
log "                    $PY -m pytest tests/"
