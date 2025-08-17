## GIMP Plugin For Integration With Meta Segment Anything

## Downloads

You can download the latest version of the plugins from the [GitHub Releases page](https://github.com/Shriinivas/gimpsegany/releases/latest).

- `gimp-segany-gimp3.zip`: For GIMP 3
- `gimp-segany-gimp2.zip`: For GIMP 2

Download the appropriate zip file and extract it in the GIMP plug-ins folder. For GIMP 3, you should extract the contents to `plug-ins/seganyplugin/`.

---

This GIMP plugin integrates with Meta's AI-based tool Segment Anything, which enables you to effortlessly isolate objects within raster images directly from GIMP.

This plugin supports both GIMP 2 and GIMP 3, using Segment Anything 1 and 2 respectively.

---

## Installation

### Plugin Installation

Please refer to the [Downloads](#downloads) section for instructions on how to download and install the plugin.

You can find GIMP's user plugin location in the `Edit > Preferences` menu under the `Folders` section. For GIMP 3, you may need to create the `plug-ins` directory if it does not already exist. Here are the default locations for each operating system:

- **Windows:** `C:\Users\[YourUsername]\AppData\Roaming\GIMP\3.0\plug-ins\`
- **Linux:** `~/.config/GIMP/3.0/plug-ins/`
- **macOS:** `~/Library/Application Support/GIMP/3.0/plug-ins/`

Make sure the plugin script (`seganyplugin.py` or `seganyplugin_GIMP2.py`) is executable.

### Segment Anything 2 Installation (for GIMP 3 Plugin)

You will get the detailed installation instructions about installing Segment Anything 2 on your platform on Meta's github site: https://github.com/facebookresearch/segment-anything-2.

**Prerequisites:**

- Python 3.10 or higher
- PyTorch 2.3.1 or higher

**Installation Steps:**

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/segment-anything-2.git
   ```
2. Navigate to the directory:
   ```bash
   cd segment-anything-2
   ```
3. Install the package:
   ```bash
   pip install -e .
   ```
4. Download a model checkpoint. There are several sizes available, such as Tiny, Small, Base Plus, and Large.

Also please ensure the `segment-anything-2` you created with `git clone` is in the PYTHONPATH. For example, if `segment-anything-2` folder is `/home/user/programs/segment-anything-2`, then your PYTHONPATH should have `/home/user/programs/segment-anything-2` included in it. You can change the .profile on linux or the corresponding file on Windows so that this is available everytime you open GIMP.

**Bridge Test (GIMP 3):**
Perform a quick check to ensure your Segment Anything 2 installation is working properly. Open a console and change direcotry to your GIMP plugin folder. Execute the following command:

```
/path/to/python3/python ./seganybridge.py sam2_hiera_large /path/to/checkpoint/model/sam2_hiera_large.pth
```

"Success!!" message in the console after running the command indicates successful installation of Segment Anything 2. Any exceptions you encounter may be resolved by referring to the Segment Anything 2 site.

### Segment Anything 1 Installation (for GIMP 2 Plugin)

You will get the detailed installation instructions about installing Segment Anything on your platform on Meta's github site: https://github.com/facebookresearch/segment-anything. There are three models or checkpoints that are published with the tool, make sure you download at least one of them (the recommended one is vit_h).

Also please ensure the `segment-anything` you created with `git clone` is in the PYTHONPATH. For example, if `segment-anything` folder is `/home/user/programs/segment-anything`, then your PYTHONPATH should have `/home/user/programs/segment-anything` included in it. You can change the .profile on linux or the corresponding file on Windows so that this is available everytime you open GIMP.

**Bridge Test (GIMP 2):**
Perform a quick check to ensure your Segment Anything installation is working properly. Open a console and change direcotry to your GIMP plugin folder. Execute the following command:

```
/path/to/python3/python ./seganybridge_SAM1.py vit_h /path/to/checkpoint/model/sam_vit_h_4b8939.pth
```

"Success!" message in the console after running the command indicates successful installation of Segment Anything. Any exceptions you encounter may be resolved by referring to the Segment Anything site.

---

## Plugin Usage

- Open GIMP. Under the "Image" menu, you should see a new submenu called "Segment Anything Layers".
- Open an image file and click on the plugin's menu item to bring up the dialog box.

**Plugin UI (GIMP 3):**

[//]: # "Add a screenshot of the GIMP 3 plugin UI here"

### Options

- **Python3 Path:** The path to the python3 instance used while running the seganybridge script.
- **Checkpoint Type / SAM2 Model Type:** The type of the Segment Anything model to use.
- **Checkpoint Path:** The path to the downloaded Segment Anything model checkpoint file.
- **Segmentation Type:** The method to be used for segmentation.
  - **Auto (GIMP 2 & 3):** Automatically segments the entire image.
  - **Box (GIMP 2 & 3):** Segments objects within a user-drawn rectangular selection.
  - **Selection (GIMP 2 & 3):** Segments objects based on sample points from a user-drawn selection.
  - **Box-Selection (GIMP 2 only):** A two-step process combining a box and selection points.
- **Mask Type:**
  - **Multiple:** Creates a separate layer for each potential object.
  - **Single:** Creates a single layer with the mask that has the highest AI probability.
- **Random Mask Color:** If checked, the generated layers will have random colors. Otherwise, a specific color can be chosen.

#### GIMP 3 Specific Options (for "Auto" Segmentation)

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
