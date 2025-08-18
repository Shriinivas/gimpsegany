## GIMP Plugin For Integration With Meta Segment Anything

## Downloads

You can download the latest version of the plugins from the [GitHub Releases page](https://github.com/Shriinivas/gimpsegany/releases/latest).

- `gimp-segany-gimp3.zip`: For GIMP 3
- `gimp-segany-gimp2.zip`: For GIMP 2

Download the appropriate zip file and extract it in the GIMP plug-ins folder. For GIMP 3, you should extract the contents to `plug-ins/seganyplugin/`.

---

This GIMP plugin integrates with Meta's AI-based tool Segment Anything, which enables you to effortlessly isolate objects within raster images directly from GIMP.

This project provides two plugins:
- **GIMP 3 Plugin**: Supports both **Segment Anything 1 (SAM1)** and **Segment Anything 2 (SAM2)**.
- **GIMP 2 Plugin**: Supports **Segment Anything 1 (SAM1)**.

---

## Installation

### Plugin Installation

Please refer to the [Downloads](#downloads) section for instructions on how to download and install the plugin.

You can find GIMP's user plugin location in the `Edit > Preferences` menu under the `Folders` section. For GIMP 3, you may need to create the `plug-ins` directory if it does not already exist. Here are the default locations for each operating system:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.0\plug-ins\`
- **Linux:** `~/.config/GIMP/3.0/plug-ins/`
- **macOS:** `~/Library/Application Support/GIMP/3.0/plug-ins/`

Make sure the plugin script (`seganyplugin.py` or `seganyplugin_GIMP2.py`) is executable.

### Segment Anything Backend Installation

You need to install the backend for the Segment Anything model you want to use. The GIMP 3 plugin can use either SAM1 or SAM2, while the GIMP 2 plugin only uses SAM1.

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

**For GIMP 3 Plugin (SAM2 model):**
```
/path/to/python3/python ./seganybridge.py sam2_hiera_large /path/to/checkpoint/model/sam2_hiera_large.pth
```

**For GIMP 3 Plugin (SAM1 model):**
```
/path/to/python3/python ./seganybridge.py vit_h /path/to/checkpoint/model/sam_vit_h_4b8939.pth
```

**For GIMP 2 Plugin (SAM1 model):**
```
/path/to/python3/python ./seganybridge_GIMP2_SAM1.py vit_h /path/to/checkpoint/model/sam_vit_h_4b8939.pth
```

A "Success!!" or "Success!" message indicates a successful installation.

---

## Plugin Usage

- Open GIMP. Under the "Image" menu, you should see a new submenu called "Segment Anything Layers".
- Open an image file and click on the plugin's menu item to bring up the dialog box.

**Plugin UI (GIMP 3):**

[//]: # "Add a screenshot of the GIMP 3 plugin UI here"

### Options

- **Python3 Path:** The path to the python3 instance used while running the seganybridge script.
- **Model Type:** The type of the Segment Anything model to use. Can be set to `Auto` to infer from the checkpoint filename (`sam_` prefix for SAM1, `sam2` for SAM2).
- **Checkpoint Path:** The path to the downloaded Segment Anything model checkpoint file (`.pth` or `.safetensors`).
- **Segmentation Type:** The method to be used for segmentation.
  - **Auto:** Automatically segments the entire image.
  - **Box:** Segments objects within a user-drawn rectangular selection.
  - **Selection:** Segments objects based on sample points from a user-drawn selection.
- **Mask Type:**
  - **Multiple:** Creates a separate layer for each potential object.
  - **Single:** Creates a single layer with the mask that has the highest AI probability.
- **Random Mask Color:** If checked, the generated layers will have random colors. Otherwise, a specific color can be chosen.

#### SAM2 Specific Options (for "Auto" Segmentation)

These options are only available for the GIMP 3 plugin when using a SAM2 model with "Auto" segmentation.

- **Segmentation Resolution:** Controls the density of the segmentation grid. Higher values will generate more masks but will be slower. (Options: Low, Medium, High)
- **Crop n Layers:** Enables segmentation on smaller, overlapping crops of the image, which can improve accuracy for smaller objects.
- **Minimum Mask Area:** Discards small, irrelevant masks.

### Workflow

1.  Select your desired options in the plugin dialog and click "OK".
2.  The plugin will create a new layer group with one or more mask layers.
3.  Find the mask layer corresponding to the object you want to isolate.
4.  Select that layer and use the "Fuzzy Selection" tool to select the mask area.
5.  Hide the new layer group and select your original image layer.
6.  You can now cut, copy, or perform any other GIMP operation on the selected object.