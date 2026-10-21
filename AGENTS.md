# gimpsegany — agent guide

GIMP 3 plugin (`seganyplugin.py`) + backend bridge (`seganybridge.py`) for
Meta Segment Anything (SAM1 / SAM2 / SAM 3.1). GIMP2 files are legacy;
leave them alone.

## Setup (do this first)

```bash
./setup.sh --help            # all options
./setup.sh --with-models --install-plugin
```

`setup.sh` creates a project-local `.venv`, installs torch, transformers,
`safetensors`, `segment-anything`, `sam2`, pytest, optionally downloads all
three checkpoints (`--models-dir`, default `~/gimpsegany-models`), and can
install the plugin into GIMP (`--plugindir` overrides autodetect).
`TORCH_INDEX_URL` selects the torch wheel index (e.g. ROCm);
`--dry-run` prints every action without doing it.

Manual equivalent: `python3 -m venv .venv`, pip-install the deps above
from `setup.sh`, point GIMP's Python3 Path at `<venv>/bin/python`.

Checkpoints:

| Backend | Source | File the bridge uses |
|---|---|---|
| SAM 3.1 | `unsloth/sam3.1` (ungated, BF16) | `model.safetensors` (any file of the folder works) |
| SAM 2.1 large | `facebook/sam2.1-hiera-large` | `sam2.1_hiera_large.pt` — the folder's `model.safetensors` is transformers-native and unloadable; the bridge says so |
| SAM1 vit_h | `https://dl.fbaipublicfiles.com/segment-anything/sam_vit_h_4b8939.pth` | the `.pth` directly |

## Verify (run these, read the output)

```bash
./test_sam3.sh [img] [python] [sam3.1-dir]   # SAM3: test, Auto, Box, Selection, guards
./test_sam12.sh [img] [python]               # SAM1+SAM2: loads, autodetect, Autos
<venv>/bin/python -m pytest tests/           # beach photo -> exactly 5 `person` masks
```

`test_sam12.sh` needs the sam12-style env (real backends); steps print real
mask geometry — a PASS label without it means nothing.

## Contracts (do not break)

- Bridge CLI is positional and backwards compatible; the plugin appends only
  `--prompt=...`. Keep `sam3.1|sam2*|vit_*|auto` accepted in arg 1.
- Backend detection lives in `loadWithFallback` (`seganybridge.py`): explicit
  type, then filename, then weight keys, then try-each (3 → 2.1 → 2 → 1).
  File evidence overrides a wrong explicit type with a printed notice.
- SAM3 Auto **requires** a text prompt; the dialog + bridge refuse without
  one. Box/Selection prompts are optional everywhere.
- SAM1 Auto caps the working image at 2048 px and upscales masks back
  (16 MP OOMs a 20 GB card otherwise); the notice print is load-bearing.
- GIMP plugin contract: one mask layer per `.seg` file, group layer named
  `Segment Anything - <type>`, selection restored at the end, error popup
  when zero masks land.
- Never commit weights, test photos, `segany_settings.json`, or caches
  (`tests/data/`, `__pycache__/` are ignored). `test_*.sh` stay out of the
  release zip (`release.yml` packages only the two plugin files).

## Debugging

- "No masks" with no popup: check the GIMP console — the bridge prints which
  backend it tried and why each attempt failed.
- Wrong-backend symptoms: `Unexpected key(s)` (2.x weights under a 2.0
  config), `not installed` (venv lacks that backend), `transformers-native`
  (SAM2 `.safetensors` instead of the `.pt`).
- The dialog stamps its build bottom-right (`PLUGIN_VERSION` in
  `seganyplugin.py`, bump on every user-visible change).
