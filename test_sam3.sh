#!/usr/bin/env bash
# Smoke test for the SAM 3 / SAM 3.1 support in seganybridge.py.
#
#   ./test_sam3.sh [image] [python] [model_dir]
#
# Runs the four bridge entry points (test mode, Auto, Box, Selection) and
# decodes the packed .seg files it writes, so each step reports real mask
# geometry. Overlay PNGs are written next to the logs for a visual check.
#
# Env overrides: PROMPT (default "bee, flower"), CUDA_VISIBLE_DEVICES
# (set it to "" to force a CPU run).

IMAGE_ARG="$1"
PY="${2:-/home/cedstrom/.venv-sam12/bin/python}"
MODEL_DIR="${3:-/mnt/fast/models/unsloth-sam3.1}"
PROMPT="${PROMPT:-bee, flower}"

WORK=/tmp/segany_sam3_test
BRIDGE="$(cd "$(dirname "$0")" && pwd)/seganybridge.py"
CKPT="$MODEL_DIR/model.safetensors"
IMG="$WORK/input.jpg"
FAILED=0
PASSED=0

log() {
    printf '%s\n' "$*"
}

fail() {
    FAILED=$((FAILED + 1))
    log "FAIL: $*"
}

pass() {
    PASSED=$((PASSED + 1))
    log "PASS: $*"
}

# run_step NAME EXPECTED_STRING -- bridge args...
run_step() {
    local name="$1"
    local expected="$2"
    shift 3
    local logfile="$WORK/$name.log"
    log ""
    log "=== $name ==="
    log "\$ $PY $BRIDGE $*"
    "$PY" "$BRIDGE" "$@" >"$logfile" 2>&1
    local code=$?
    if [ $code -ne 0 ]; then
        fail "$name exited with code $code"
        tail -20 "$logfile"
        return 1
    fi
    if [ -n "$expected" ] && ! grep -q "$expected" "$logfile"; then
        fail "$name: expected '$expected' in the output"
        tail -20 "$logfile"
        return 1
    fi
    grep -v -e 'it/s\]' -e '^$' -e '(null)' "$logfile" | tail -8
    pass "$name"
    return 0
}

# run_bridge NAME EXPECTED [bridge args after model type and checkpoint]
run_bridge() {
    local name="$1"
    local expected="$2"
    shift 2
    run_step "$name" "$expected" -- sam3.1 "$CKPT" "$@"
}

# summarize PREFIX IMAGE -> per mask: pixels, fraction, bbox; writes overlays
summarize() {
    local prefix="$1"
    local image="$2"
    "$PY" - "$prefix" "$image" <<'PYEOF'
import os
import sys
import numpy as np
import cv2

prefix, image = sys.argv[1], sys.argv[2]
base = cv2.imread(image)
count = 0
while os.path.exists(f"{prefix}{count}.seg"):
    with open(f"{prefix}{count}.seg", "rb") as f:
        data = f.read()
    rows = int.from_bytes(data[0:4], "big")
    cols = int.from_bytes(data[4:8], "big")
    bits = np.unpackbits(
        np.frombuffer(data[8:], dtype=np.uint8), bitorder="little"
    )
    mask = bits[: rows * cols].astype(bool).reshape(rows, cols)
    ys, xs = np.nonzero(mask)
    bbox = (
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if len(ys)
        else None
    )
    print(
        f"  mask {count}: {mask.shape} pixels={int(mask.sum())} "
        f"frac={mask.sum() / mask.size:.4f} bbox={bbox}"
    )
    if base is not None and base.shape[:2] == mask.shape:
        overlay = base.copy()
        overlay[mask] = (
            0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)
        cv2.imwrite(f"{prefix}{count}_overlay.png", overlay)
    count += 1
print(f"  total masks: {count}")
if count == 0:
    sys.exit(1)
PYEOF
}

check_masks() {
    local name="$1"
    local prefix="$2"
    if summarize "$prefix" "$IMG"; then
        pass "$name produced masks"
    else
        fail "$name produced no masks"
    fi
}

mkdir -p "$WORK"

if [ ! -x "$PY" ]; then
    log "Python not found: $PY (pass it as the 2nd argument)"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    log "Model not found: $CKPT"
    log "Download it with: hf download unsloth/sam3.1 --local-dir $MODEL_DIR"
    exit 1
fi

# Test image: use the given file, else fetch the HF bee photo.
src="$IMAGE_ARG"
if [ -n "$src" ] && [ ! -f "$src" ]; then
    log "Image not found: $src"
    exit 1
fi
if [ -z "$src" ]; then
    src="$WORK/bee.jpg"
    if [ ! -f "$src" ]; then
        log "Downloading test image ..."
        curl -sfL -o "$src" \
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" || {
            fail "could not download the test image"
            exit 1
        }
    fi
fi
"$PY" - "$src" "$IMG" <<'PYEOF'
import sys
import cv2
image = cv2.imread(sys.argv[1])
if image is None:
    sys.exit("cannot read " + sys.argv[1])
scale = 512 / image.shape[1]
small = cv2.resize(image, (512, int(image.shape[0] * scale)))
cv2.imwrite(sys.argv[2], small)
print(f"test image {sys.argv[2]}: {small.shape}")
PYEOF

"$PY" -c "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

W=$("$PY" -c "import cv2,sys;print(cv2.imread('$IMG').shape[1])")
H=$("$PY" -c "import cv2,sys;print(cv2.imread('$IMG').shape[0])")
BOX="$((W / 4)),$((H / 4)),$((3 * W / 4)),$((3 * H / 4))"
printf '%s %s\n%s %s\n%s %s\n%s %s\n' \
    "$((W / 2 - 40))" "$((H / 2 - 40))" "$((W / 2 + 40))" "$((H / 2 - 40))" \
    "$((W / 2 + 40))" "$((H / 2 + 40))" "$((W / 2 - 40))" "$((H / 2 + 40))" \
    >"$WORK/sel.txt"

log "prompt: '$PROMPT'   box: $BOX   image: ${W}x${H}"

# 1. Test mode (loads the model, one tiny forward pass)
run_bridge test_mode "Success!!"

# 2. Auto: every instance of the prompted concepts
rm -f "$WORK/auto_"*.seg
run_bridge auto "SAM3 wrote" "$IMG" Auto Multiple "$WORK/auto_" True Medium 0 0 "--prompt=$PROMPT"
check_masks auto "$WORK/auto_"

# 3. Box: the GIMP selection bounding box as a visual prompt
rm -f "$WORK/box_"*.seg
run_bridge box "SAM3 wrote" "$IMG" Box Multiple "$WORK/box_" True sel_place_holder "$BOX" "--prompt=$PROMPT"
check_masks box "$WORK/box_"

# 4. Selection: exported sample points -> bounding box prompt
rm -f "$WORK/sel_"*.seg
run_bridge selection "SAM3 wrote" "$IMG" Selection Multiple "$WORK/sel_" True "$WORK/sel.txt"
check_masks selection "$WORK/sel_"

# 5. Auto with no prompt must refuse instead of guessing "object"
rm -f "$WORK/noprompt_"*.seg
run_bridge empty_prompt "needs a text prompt" "$IMG" Auto Multiple "$WORK/noprompt_" True Medium 0 0
if compgen -G "$WORK/noprompt_*.seg" > /dev/null; then
    fail "empty_prompt wrote masks despite refusing"
else
    pass "empty_prompt wrote no masks"
fi

# 6. The SAM1 backend must be present and attempted (dummy file, real load try).
: >"$WORK/sam_vit_h_4b8939.pth"
run_step sam1_backend_present "Error loading SAM1 model" -- vit_h "$WORK/sam_vit_h_4b8939.pth"

log ""
log "======================================"
log "passed: $PASSED   failed: $FAILED"
log "overlays: $WORK/{auto,box,sel}_N_overlay.png"
log "logs:     $WORK/*.log"
log "======================================"
[ "$FAILED" -eq 0 ]
