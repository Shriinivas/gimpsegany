"""Shared fixtures for the gimpsegany pytest suite."""
import os
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BEACH_URL = (
    "https://images.unsplash.com/photo-1517486808906-6ca8b3f04846"
    "?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb"
    "&dl=naassom-azevedo-Q_Sei-TqSlc-unsplash.jpg"
)
BEACH_NAME = "naassom-azevedo-Q_Sei-TqSlc-unsplash.jpg"
MODEL_DIR = os.environ.get("SAM3_MODEL_DIR", "/mnt/fast/models/unsloth-sam3.1")


@pytest.fixture(scope="session")
def beach_jpg():
    """The Unsplash beach photo, downloaded once and cached in tests/data/."""
    data_dir = REPO_ROOT / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / BEACH_NAME
    if not dest.exists():
        urllib.request.urlretrieve(BEACH_URL, dest)
    assert dest.stat().st_size > 0, f"downloaded image is empty: {dest}"
    return dest


@pytest.fixture(scope="session")
def sam3():
    """A loaded SAM3 model handle, shared by the whole session (slow to load)."""
    import torch

    import seganybridge

    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"SAM3 model dir not found: {MODEL_DIR}")
    strategy = seganybridge.SAM3Strategy()
    handle = strategy.load_model(
        os.path.join(MODEL_DIR, "model.safetensors"), "sam3.1"
    )
    if handle is None:
        pytest.fail("SAM3 model failed to load")
    if torch.cuda.is_available():
        handle.to("cuda")
    return strategy, handle
