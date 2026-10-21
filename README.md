## GIMP Plugin For Integration With Meta Segment Anything

## Downloads

You can download the latest version of the plugin from the [GitHub Releases page](https://github.com/Shriinivas/gimpsegany/releases/latest).

- `gimp-segany-gimp3.zip`: For GIMP 3

Download the zip file and follow the installation instructions below.

---

This GIMP plugin integrates with Meta's AI-based tool Segment Anything, which enables you to effortlessly isolate objects within raster images directly from GIMP.

This project provides a plugin that supports **Segment Anything 3 (SAM3)**, **Segment Anything 2 (SAM2)** and **Segment Anything 1 (SAM1)**.

---

## Installation

### Plugin Installation

1.  Go to the [GitHub Releases page](https://github.com/Shriinivas/gimpsegany/releases/latest) and download `gimp-segany-gimp3.zip`.
2.  Extract the zip file in your GIMP `plug-ins` folder. This will create a `seganyplugin` directory with the plugin files inside.
3.  If you are updating, you can extract and replace the existing `seganyplugin` directory.

You can find GIMP's user plugin location in the `Edit > Preferences` menu under the `Folders` section. Here are the default locations for each operating system:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.0\plug-ins\`
- **Linux:** `~/.config/GIMP/3.0/plug-ins/`
- **macOS:** `~/Library/Application Support/GIMP/3.0/plug-ins/`

After extracting, ensure the plugin script is executable (on Linux and macOS).

### Segment Anything Backend Installation

You need to install the backend for the Segment Anything model you want to use. The plugin can use SAM3 (SAM 3 / SAM 3.1), SAM2 (SAM 2 / SAM 2.1) and SAM1.

Quick start (agents: see `AGENTS.md`):

```bash
./setup.sh --with-models --install-plugin   # venv + deps + checkpoints + GIMP install
```

`./setup.sh --help` lists every option (`--venv`, `--models-dir`, `--plugindir`,
`--dry-run`); `TORCH_INDEX_URL` selects the torch wheel index for ROCm systems.

#### Segment Anything 3 (SAM3 / SAM 3.1) Installation

The plugin runs SAM 3 and SAM 3.1 through the [Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/sam3) port of the model, so any `Sam3Model` checkpoint folder works. The ungated mirror [unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1) is the easiest to use: it needs no access request, and its BF16 weights run on NVIDIA and on AMD GPUs (ROCm).

**Prerequisites:**

- Python 3.10 or higher (3.12+ recommended)
- PyTorch 2.7 or higher with a CUDA or ROCm build

**Installation Steps:**

1. Install the model library into the Python environment the plugin will run:
   ```bash
   pip install "transformers>=5.10" pillow opencv-python-headless
   ```
2. Download the checkpoint (about 1.7 GB):
   ```bash
   hf download unsloth/sam3.1 --local-dir /path/to/sam3.1
   ```
   Keep `config.json`, `model.safetensors` and the tokenizer files in the same folder.
3. In the plugin dialog select `model.safetensors` (or any file of that folder) as **Checkpoint Path**. The model backend is detected from the checkpoint automatically.

**AMD / ROCm:** install a ROCm PyTorch build into the same environment, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

The FP8 and INT8 files of `unsloth/sam3.1` require quantization kernels that are not available on ROCm, so use the BF16 `model.safetensors`.

#### Segment Anything 2 (SAM2) Installation

You will get the detailed installation instructions about installing Segment Anything 2 on your platform on Meta's github site: https://github.com/facebookresearch/segment-anything-2.

**Prerequisites:**

- Python 3.10 or higher
- PyTorch 2.3.1 or higher

**Installation Steps:**

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/segment-anything-2.git
   ```
2. Navigate to the directory and install the package:
   ```bash
   cd segment-anything-2 && pip install -e .
   ```
3. Download a model checkpoint (e.g., Tiny, Small, Base Plus, Large).
4. Ensure the `segment-anything-2` directory is in your `PYTHONPATH`.

#### Segment Anything 1 (SAM1) Installation

You will get the detailed installation instructions about installing Segment Anything on your platform on Meta's github site: https://github.com/facebookresearch/segment-anything.

**Installation Steps:**

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/segment-anything.git
   ```
2. Navigate to the directory and install the package:
   ```bash
   cd segment-anything && pip install -e .
   ```
3. Download a model checkpoint (e.g., `vit_h`, `vit_l`, `vit_b`).
4. Ensure the `segment-anything` directory is in your `PYTHONPATH`.

### Bridge Test

Perform a quick check to ensure your Segment Anything installation is working properly. Open a console and change directory to your GIMP plugin folder.

**For a SAM3 model (SAM 3 / SAM 3.1):**

```
/path/to/python3/python ./seganybridge.py sam3.1 /path/to/sam3.1/model.safetensors
```

A scripted end-to-end check of the SAM3 path (test mode, Auto, Box and Selection, plus the geometry of every mask written) is in `test_sam3.sh`:

```
./test_sam3.sh [image] [path/to/python3/python] [path/to/sam3.1]
```

**For SAM2 model:**

```
/path/to/python3/python ./seganybridge.py sam2_hiera_large /path/to/checkpoint/model/sam2_hiera_large.pth
```

**For SAM1 model:**

```
/path/to/python3/python ./seganybridge.py vit_h /path/to/checkpoint/model/sam_vit_h_4b8939.pth
```

Any explicit type also works; `auto` in its place detects the backend from the checkpoint.

---

## Plugin Usage

- Open GIMP. Under the "Image" menu, you should see a new submenu called "Segment Anything Layers".
- Open an image file and click on the plugin's menu item to bring up the dialog box.

**Plugin UI:**

[//]: # "Add a screenshot of the GIMP 3 plugin UI here"

### Options

- **Python3 Path:** The path to the python3 instance used while running the seganybridge script.
- **Checkpoint Path:** The path to the downloaded Segment Anything model checkpoint file (`.pth`, `.safetensors`, or a SAM3 model folder). The backend, version and size are detected from the checkpoint automatically: an explicit type is tried first, then the filename, then the weights themselves, then every backend in turn (SAM3, 2.1 sizes, 2.0 sizes, SAM1 sizes) until one loads.
- **Segmentation Type:** The method to be used for segmentation.
  - **Auto:** Automatically segments the entire image.
  - **Box:** Segments objects within a user-drawn rectangular selection.
  - **Selection:** Segments objects based on sample points from a user-drawn selection.
- **Mask Type:**
  - **Multiple:** Creates a separate layer for each potential object.
  - **Single:** Creates a single layer with the mask that has the highest AI probability.
- **Text Prompt (SAM3):** One or more comma separated noun phrases (`sofa, lamp`) used to prompt a SAM 3 model. Visible only for SAM3 models.
- **Random Mask Color:** If checked, the generated layers will have random colors. Otherwise, a specific color can be chosen.

#### SAM3 Specific Options

These options apply when a SAM 3 / SAM 3.1 model is used.

- **Text Prompt:** SAM 3 is a promptable concept detector: it segments every instance of the concept you name. Enter one or more short noun phrases separated by commas, for example `person, red car`. The prompt is required for "Auto" segmentation and optional for "Box" and "Selection", where it refines what the visual prompt selects.
- **Minimum Mask Area:** works the same way as for SAM2.
- **Selection:** the Transformers port of SAM 3 takes box prompts, not point prompts, so the bounding box of your GIMP selection is used as the visual prompt.
- **Segmentation Resolution** and **Crop n Layers** are SAM2 only and are hidden for SAM3.


#### SAM1 Specific Behavior

- **Auto** downscales images whose longest side exceeds 2048 px to fit GPU
  memory (a 16 MP photo otherwise exhausts even a 20 GB card) and upscales
  the masks back to full size. A notice is printed when this kicks in. Box
  and Selection run at full resolution.

#### SAM2 Specific Options (for "Auto" Segmentation)

These options are only available when using a SAM2 model with "Auto" segmentation.

- **Segmentation Resolution:** Controls the density of the segmentation grid. Higher values will generate more masks but will be slower. (Options: Low, Medium, High)
- **Crop n Layers:** Enables segmentation on smaller, overlapping crops of the image, which can improve accuracy for smaller objects.
- **Minimum Mask Area:** Discards small, irrelevant masks.
- Like SAM1, **Auto** downscales images whose longest side exceeds 2048 px
  to fit GPU memory (High density on a 16 MP photo otherwise exhausts a
  20 GB card) and upscales the masks back to full size.

### Which backend for large photos

Auto memory usage does **not** improve much from SAM1 to SAM2 — both build
dense full-resolution proposals, and both exhaust a 20 GB card on a 16 MP
photo (SAM1: 14.5 GB held + 11.5 GB attempted; SAM2 Medium: 13 GB + 6.7 GB).
SAM2 is somewhat leaner and faster at the same working size, and High
density quadruples its proposals. SAM3 is the step-change: no proposal
grid, so a 16 MP photo segments at full resolution with no downscaling.

| Backend (Auto) | 16 MP full-res | At 2048 px cap |
|---|---|---|
| SAM1 | OOM | 74 masks, ~5 s, 7.7 GB peak |
| SAM2 Medium, min area 100 | OOM | 42 masks, ~3 s, 5.1 GB peak |
| SAM2 High, min area 100 | OOM | 62 masks, ~7 s, 5.3 GB peak |
| SAM3 (`person`) | 5 masks, full resolution, no downscale | n/a |

Measured on a 20 GB AMD card (ROCm); SAM3 in BF16, SAM1/SAM2 in FP32.
Rule of thumb: SAM3 for big photos when a text prompt fits the task;
SAM1/SAM2 Auto is capped at 2048 px regardless.

### Workflow

1.  Select your desired options in the plugin dialog and click "OK".
2.  The plugin will create a new layer group with one or more mask layers.
3.  Find the mask layer corresponding to the object you want to isolate.
4.  Select that layer and use the "Fuzzy Selection" tool to select the mask area.
5.  Hide the new layer group and select your original image layer.
6.  You can now cut, copy, or perform any other GIMP operation on the selected object.
