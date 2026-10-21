"""SAM3 must extract the 5 people from the Unsplash beach photo.

Image: https://images.unsplash.com/photo-1517486808906-6ca8b3f04846
(portrait photo, five people wading in the lower half).
"""
import cv2
import numpy as np

PROMPT = "person"
EXPECTED_PEOPLE = 5


def _decode_seg(path):
    with open(path, "rb") as f:
        data = f.read()
    rows = int.from_bytes(data[0:4], "big")
    cols = int.from_bytes(data[4:8], "big")
    bits = np.unpackbits(np.frombuffer(data[8:], dtype=np.uint8), bitorder="little")
    return bits[: rows * cols].reshape(rows, cols).astype(bool)


def test_auto_person_extracts_five_people(sam3, beach_jpg, tmp_path):
    strategy, handle = sam3
    bgr = cv2.imread(str(beach_jpg))
    assert bgr is not None, f"could not read {beach_jpg}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    prefix = str(tmp_path / "person_")
    strategy.segment_auto(
        handle,
        rgb,
        prefix,
        True,
        maskType="Multiple",
        textPrompt=PROMPT,
        minMaskArea=0,
    )

    found = sorted(tmp_path.glob("person_*.seg"))
    assert len(found) == EXPECTED_PEOPLE, (
        f"expected {EXPECTED_PEOPLE} '{PROMPT}' masks, got {len(found)}"
    )

    height, width = rgb.shape[:2]
    for mask_file in found:
        mask = _decode_seg(mask_file)
        assert mask.shape == (height, width), mask_file.name
        frac = mask.mean()
        assert 0.01 < frac < 0.30, f"{mask_file.name}: frac={frac:.4f}"
        ys, _ = np.nonzero(mask)
        assert ys.mean() > 0.45 * height, f"{mask_file.name} is not in the lower half"
