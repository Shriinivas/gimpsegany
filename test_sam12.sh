#!/usr/bin/env bash
# Smoke test for the SAM1 / SAM2 paths in seganybridge.py, including the
# content-based autodetect (2.0 vs 2.1, family and vit size from weights).
#
#   ./test_sam12.sh [image] [python]
#
# Needs an env with the `segment_anything` and `sam2` backends, e.g.
# /home/cedstrom/.venv-sam12/bin/python, and the checkpoints below.

IMAGE_ARG="$1"
PY="${2:-/home/cedstrom/.venv-sam12/bin/python}"
VIT_H="/mnt/fast/models/sam_vit_h_4b8939.pth"
S21_PT="/mnt/fast/models/sam2.1-hiera-large/sam2.1_hiera_large.pt"
S21_SF="/mnt/fast/models/sam2.1-hiera-large/model.safetensors"

WORK=/tmp/segany_sam12_test
BRIDGE="$(cd "$(dirname "$0")" && pwd)/seganybridge.py"
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

# run_step NAME EXPECTED -- bridge args...
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
    local image="$3"
    if summarize "$prefix" "$image"; then
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
for f in "$VIT_H" "$S21_PT" "$S21_SF"; do
    if [ ! -f "$f" ]; then
        log "Checkpoint not found: $f"
        exit 1
    fi
done

# Test image: use the given file, else the repo beach photo, resized to 512 px.
src="$IMAGE_ARG"
if [ -n "$src" ] && [ ! -f "$src" ]; then
    log "Image not found: $src"
    exit 1
fi
if [ -z "$src" ]; then
    src="$(cd "$(dirname "$0")" && pwd)/naassom-azevedo-Q_Sei-TqSlc-unsplash.jpg"
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
IMGFULL="$src"

"$PY" -c "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 1-3. Test mode: each backend loads its checkpoint (bare auto throughout).
run_step vit_h_load "Success!!" -- auto "$VIT_H"
run_step sam21_pt_load "Success!!" -- auto "$S21_PT"
# The HF model.safetensors is transformers-native (Sam2VideoModel): the sam2
# package cannot load it. Point the user at the Meta-layout .pt instead.
run_step sam21_safetensors_native "transformers-native" -- sam2.1_hiera_large "$S21_SF"
# Bare auto hits the same guard (the sweep only applies to loadable files).
run_step sam21_safetensors_auto "transformers-native" -- auto "$S21_SF"
# A Meta-layout .safetensors (converted from the .pt here) loads fine and
# exercises the safetensors -> pth conversion.
S21_CONV="$WORK/sam21_meta.safetensors"
if [ ! -f "$S21_CONV" ]; then
    "$PY" -c "from safetensors.torch import save_file; import torch; save_file(torch.load('$S21_PT', map_location='cpu', weights_only=True)['model'], '$S21_CONV')"
fi
run_step sam21_safetensors_load "Success!!" -- auto "$S21_CONV"

# 4. A wrong explicit 2.0 type with 2.1 weights must follow the file.
run_step sam21_version_correction "using config for 'sam2.1_hiera_large'" -- sam2_hiera_large "$S21_PT"

# 5. Renamed SAM1 file: size comes from the encoder depth in the weights.
ln -sf "$VIT_H" "$WORK/weights.bin"
run_step sam1_renamed "from weights: vit_h" -- auto "$WORK/weights.bin"

# 6. Renamed SAM2 file: version comes from the keys, size via sweep.
ln -sf "$S21_PT" "$WORK/other.bin"
run_step sam2_renamed "Loaded with sam2.1_hiera_large" -- auto "$WORK/other.bin"

# 7. A garbage file tries everything, then fails with the attempt list.
: > "$WORK/junk.bin"
run_step junk_refused "could not load" -- auto "$WORK/junk.bin"

# 8. SAM 2.1 Auto segmentation produces masks.
rm -f "$WORK/s21auto_"*.seg
run_step sam21_auto "SAM2" -- auto "$S21_PT" "$IMG" Auto Multiple "$WORK/s21auto_" True Medium 0 0
check_masks sam21_auto "$WORK/s21auto_" "$IMG"

# 9. SAM1 Auto segmentation produces masks (SAM1 omits the SAM2-only args).
rm -f "$WORK/vitauto_"*.seg
run_step vit_h_auto "SAM1 wrote" -- auto "$VIT_H" "$IMG" Auto Multiple "$WORK/vitauto_" True
check_masks vit_h_auto "$WORK/vitauto_" "$IMG"

# 10. SAM1 Auto on the full-size photo (downscale + upscale path).
rm -f "$WORK/vitfull_"*.seg
run_step vit_h_auto_full "SAM1 wrote" -- auto "$VIT_H" "$IMGFULL" Auto Multiple "$WORK/vitfull_" True
if grep -q "downscaled" "$WORK/vit_h_auto_full.log"; then
    pass "vit_h_auto_full used the downscale path"
else
    fail "vit_h_auto_full did not print the downscale notice"
fi
check_masks vit_h_auto_full "$WORK/vitfull_" "$IMGFULL"

# 11. SAM2 Auto High on the full-size photo (the reported X-crash config:
# High density + min area 100 at 16 MP OOMed before the downscale cap).
rm -f "$WORK/s21hires_"*.seg
run_step sam21_hires "SAM2 wrote" -- auto "$S21_PT" "$IMGFULL" Auto Multiple "$WORK/s21hires_" True High 0 100
if grep -q "downscaled" "$WORK/sam21_hires.log"; then
    pass "sam21_hires used the downscale path"
else
    fail "sam21_hires did not print the downscale notice"
fi
check_masks sam21_hires "$WORK/s21hires_" "$IMGFULL"

log ""
log "======================================"
log "passed: $PASSED   failed: $FAILED"
log "overlays: $WORK/{s21auto,vitauto}_N_overlay.png"
log "logs:     $WORK/*.log"
log "======================================"
[ "$FAILED" -eq 0 ]
