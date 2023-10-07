## GIMP Plugin For Integration With Meta Segment Anything
This GIMP plugin integrates with Meta's AI-based tool Segment Anything,  which enables you to effortlessly isolate objects within raster images directly from GIMP. 

### Installation I - Segment Anything local instance
You will get the detailed installation instructions about installing Segment Anything on your platform on Meta's github site: https://github.com/facebookresearch/segment-anything. There are three models or checkpoints that are published with the tool, make sure you download at least one of them (the recommended one is vit_h).

### Installation II - Segment Anything GIMP plugin
- Make sure your GIMP version supports Python-Fu. Open GIMP and check for the "Python-Fu" submenu under "Filters". If you don't see it, you'll need to install a Python-enabled version of GIMP on your machine. Arch Linux users can find an AUR package called "python2-gimp" for this purpose.
- Once you have Python-GIMP installed, download the zip file from the plugin's GitHub location:https://github.com/Shriinivas/gimpsegany and extract the files: seganyplugin.py and seganybridge.py into the GIMP plugin folder. You can find GIMP's user plugin location in the Edit-Preferences menu under the Folders section.
- Note down the following information:
  - The python3 instance used while running the seganybridge script. In case you're not using any special environment, this will be the default python3 instance. If you've set up a separate environment for Segment Anything - for example pyenv or conda - consult its documentation to determine the python3 path.
  - The location of the checkpoint files that you downloaded, which you want Segment Anything to use
- Perform a quick check to ensure your Segment Anything installation is working properly. Open console and change direcotry to gimp plugin-folder. Execute the following command:
    ```
    /path/to/python3/python ./seganybridge.py vit_h /path/to/checkpoint/model/sam_vit_h_4b8939.pth
    ```
    "Success!" message in the console after running the command indicates successful installation of Segment Anything. Any exceptions you encounter may be resolved by referring to the Segment Anything site.

### Plugin Usage
- Open GIMP. Under the "Image" menu, you should see a new submenu called "Segment Anything Layers." 
- Open an image file
- Click on "Segment Anything Layers," which will bring up the following dialog box
- 
![image](https://github.com/Shriinivas/gimpsegany/assets/42069100/b90c67bc-1529-4bd3-8df0-950e45e1e871)

  - Choose the Python instance path discussed earlier in the 'Python3 Path' file chooser. 
  - Select the checkpoint type (e. g. vit_h) from the Checkpoint Type dropdown
  - Select the corresponding checkpoint file in Checkpoint Path fie chooser
  - Choose one of the four Segmentation Types (explanation below)
  - If you check the Random Mask Color the generated layers will have random color mask regions. Otherwise you can choose a specific color for mask regions in the new layers
  - In case of Segmentation Type other than Auto, there will be another dropdown allowing you to choose the mask type. With "Multiple" more than one layer will be created and with "Single" only one layer with mask having maximum AI probability is created. 
  - Click "OK"
  - The plugin creates a new layer group with one or more layers, each will have a specific region filled with mask color representing a potential object.
  - Find the mask layer corresponding to the object you want to isolate.
  - Select that layer and use the "Fuzzy Selection" tool.
  - Click anywhere within the mask area, optionally adding a one or two-pixel feather for smoothness. This will select the entire area which corresponds to the object.
  - Now hide the newly created layer group, select the image layer
  - You can cut the selection area and paste it as new layer. And by hiding the image layer now you get the desired object isolated.
  - Alternately you can perform GIMP image operations, like colorization, on the image selection (the object)
  
#### Segmentation Types
- Auto: The "Auto" Segmentation Type lets Segment Anything decide the objects automatically. With this type selected, a number of layers are created since Segment Anything tries to isolate every possible object. If you're running GIMP via the console, you can see the commands getting executed and the layers being created.
- Box: To isolate a specific object without segmenting everything, draw a selection box around the object you want to isolate. This will restrict the search area for segmentation objects to the selected box
- Selection: The "Selection" type lets you choose an arbitrary area on the image. The tool passes sample points (the count specified in "Selection Points" edit box). Segment Anything then attempts to identify the object based on these points.
- Box-Selection: Is a combination of Box and Selection types. It involves a two-step process. First, select a rectangular area, convert it to a path. Then create another selection containing the sample points. Invoke the plugin, choose "Box-Selection" as the Segmentation Type, and, in addition to Selection Points, select the path that corresponds to the rectangular selection.The resulting layers will include colored regions related to the objects within the box, which are associated with the sample points.

### [Detailed Video Tutorial](https://youtu.be/xyuSe0SaMHk)
