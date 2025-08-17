#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp

gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi

gi.require_version("Gdk", "3.0")
from gi.repository import Gdk

gi.require_version("Gegl", "0.4")
from gi.repository import GObject, Gio, Gegl
from gi.repository import GLib


import tempfile
import subprocess
from os.path import exists
from array import array
import random
import os
import sys
import glob
import struct
import json
import logging

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


class DialogValue:
    def __init__(self, filepath):
        data = None
        self.pythonPath = None
        self.modelType = "sam2_hiera_large"
        self.checkPtPath = None
        self.maskType = "Multiple"
        self.segType = "Auto"
        self.isRandomColor = False
        self.maskColor = [255, 0, 0, 255]
        self.selPtCnt = 10
        self.selBoxPathName = None
        self.segRes = "Medium"
        self.cropNLayers = 0
        self.minMaskArea = 0

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.pythonPath = data.get("pythonPath", self.pythonPath)
                self.modelType = data.get("modelType", self.modelType)
                self.checkPtPath = data.get("checkPtPath", self.checkPtPath)
                self.maskType = data.get("maskType", self.maskType)
                self.segType = data.get("segType", self.segType)
                self.isRandomColor = data.get("isRandomColor", self.isRandomColor)
                self.maskColor = data.get("maskColor", self.maskColor)
                self.selPtCnt = data.get("selPtCnt", self.selPtCnt)
                self.segRes = data.get("segRes", self.segRes)
                self.cropNLayers = data.get("cropNLayers", self.cropNLayers)
                self.minMaskArea = data.get("minMaskArea", self.minMaskArea)
        except Exception as e:
            logging.info("Error reading json : %s" % e)

    def persist(self, filepath):
        data = {
            "pythonPath": self.pythonPath,
            "modelType": self.modelType,
            "checkPtPath": self.checkPtPath,
            "maskType": self.maskType,
            "segType": self.segType,
            "isRandomColor": self.isRandomColor,
            "maskColor": self.maskColor,
            "selPtCnt": self.selPtCnt,
            "segRes": self.segRes,
            "cropNLayers": self.cropNLayers,
            "minMaskArea": self.minMaskArea,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)


class OptionsDialog(Gtk.Dialog):
    def __init__(self, image, boxPathDict):
        Gtk.Dialog.__init__(
            self, title="Segment Anything 2", transient_for=None, flags=0
        )
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK
        )

        self.set_default_size(400, 200)

        self.boxPathNames = sorted(boxPathDict.keys())
        boxPathExist = len(self.boxPathNames) > 0
        isGrayScale = image.get_base_type() == Gimp.ImageType.GRAYA_IMAGE
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        self.configFilePath = os.path.join(scriptDir, "segany_settings.json")

        self.values = DialogValue(self.configFilePath)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        self.get_content_area().add(grid)

        # Python Path
        pythonFileLbl = Gtk.Label(label="Python3 Path:", xalign=1)
        self.pythonFileBtn = Gtk.FileChooserButton(title="Select Python Path")
        if self.values.pythonPath is not None:
            self.pythonFileBtn.set_filename(self.values.pythonPath)
        grid.attach(pythonFileLbl, 0, 0, 1, 1)
        grid.attach(self.pythonFileBtn, 1, 0, 1, 1)

        # Checkpoint Type
        modelTypeLbl = Gtk.Label(label="SAM2 Model Type:", xalign=1)
        self.modelTypeDropDown = Gtk.ComboBoxText()
        self.modelTypeVals = [
            "sam2_hiera_large",
            "sam2_hiera_base_plus",
            "sam2_hiera_small",
            "sam2_hiera_tiny",
        ]
        for value in self.modelTypeVals:
            self.modelTypeDropDown.append_text(value)
        self.modelTypeDropDown.set_active(
            self.modelTypeVals.index(self.values.modelType)
        )
        grid.attach(modelTypeLbl, 0, 1, 1, 1)
        grid.attach(self.modelTypeDropDown, 1, 1, 1, 1)

        # Checkpoint Path
        checkPtFileLbl = Gtk.Label(
            label="SAM2 Checkpoint (.pt/.safetensors):", xalign=1
        )
        self.checkPtFileBtn = Gtk.FileChooserButton(title="Select SAM2 Checkpoint Path")
        if self.values.checkPtPath is not None:
            self.checkPtFileBtn.set_filename(self.values.checkPtPath)
        grid.attach(checkPtFileLbl, 0, 2, 1, 1)
        grid.attach(self.checkPtFileBtn, 1, 2, 1, 1)

        # Segmentation Type
        segTypeLbl = Gtk.Label(label="Segmentation Type:", xalign=1)
        self.segTypeDropDown = Gtk.ComboBoxText()
        self.segTypeVals = ["Auto", "Box", "Selection"]
        for value in self.segTypeVals:
            self.segTypeDropDown.append_text(value)
        self.segTypeDropDown.set_active(self.segTypeVals.index(self.values.segType))
        grid.attach(segTypeLbl, 0, 3, 1, 1)
        grid.attach(self.segTypeDropDown, 1, 3, 1, 1)

        # Mask Type
        self.maskTypeLbl = Gtk.Label(label="Mask Type:", xalign=1)
        self.maskTypeDropDown = Gtk.ComboBoxText()
        self.maskTypeVals = ["Multiple", "Single"]
        for value in self.maskTypeVals:
            self.maskTypeDropDown.append_text(value)
        self.maskTypeDropDown.set_active(self.maskTypeVals.index(self.values.maskType))
        grid.attach(self.maskTypeLbl, 0, 4, 1, 1)
        grid.attach(self.maskTypeDropDown, 1, 4, 1, 1)

        # Selection Points
        self.selPtsLbl = Gtk.Label(label="Selection Points:", xalign=1)
        self.selPtsEntry = Gtk.Entry()
        self.selPtsEntry.set_text(str(self.values.selPtCnt))
        grid.attach(self.selPtsLbl, 0, 5, 1, 1)
        grid.attach(self.selPtsEntry, 1, 5, 1, 1)

        # ... (previous UI elements)

        # Segmentation Resolution
        self.segResLbl = Gtk.Label(label="Segmentation Resolution:", xalign=1)
        self.segResDropDown = Gtk.ComboBoxText()
        self.segResVals = ["Low", "Medium", "High"]
        for value in self.segResVals:
            self.segResDropDown.append_text(value)
        self.segResDropDown.set_active(self.segResVals.index(self.values.segRes))
        grid.attach(self.segResLbl, 0, 6, 1, 1)
        grid.attach(self.segResDropDown, 1, 6, 1, 1)

        # Crop n Layers
        self.cropNLayersLbl = Gtk.Label(label="Crop n Layers:", xalign=1)
        self.cropNLayersChk = Gtk.CheckButton()
        self.cropNLayersChk.set_active(self.values.cropNLayers > 0)
        grid.attach(self.cropNLayersLbl, 0, 7, 1, 1)
        grid.attach(self.cropNLayersChk, 1, 7, 1, 1)

        # Minimum Mask Area
        self.minMaskAreaLbl = Gtk.Label(label="Minimum Mask Area:", xalign=1)
        self.minMaskAreaEntry = Gtk.Entry()
        self.minMaskAreaEntry.set_text(str(self.values.minMaskArea))
        grid.attach(self.minMaskAreaLbl, 0, 8, 1, 1)
        grid.attach(self.minMaskAreaEntry, 1, 8, 1, 1)

        # Mask Color
        if not isGrayScale:
            self.randColBtn = Gtk.CheckButton(label="Random Mask Color")
            self.randColBtn.set_active(self.values.isRandomColor)
            grid.attach(self.randColBtn, 1, 9, 1, 1)

            self.maskColorLbl = Gtk.Label(label="Mask Color:", xalign=1)
            self.maskColorBtn = Gtk.ColorButton()
            rgba = Gdk.RGBA()
            rgba.parse(
                f"rgb({self.values.maskColor[0]},{self.values.maskColor[1]},{self.values.maskColor[2]})"
            )
            self.maskColorBtn.set_rgba(rgba)
            grid.attach(self.maskColorLbl, 0, 10, 1, 1)
            grid.attach(self.maskColorBtn, 1, 10, 1, 1)

        self.segTypeDropDown.connect("changed", self.on_seg_type_changed)
        if not isGrayScale:
            self.randColBtn.connect("toggled", self.on_random_toggled)

        self.on_seg_type_changed(self.segTypeDropDown)
        if not isGrayScale:
            self.on_random_toggled(self.randColBtn)

        self.show_all()

    def on_seg_type_changed(self, widget):
        segType = self.segTypeVals[self.segTypeDropDown.get_active()]
        isAuto = segType == "Auto"
        self.selPtsLbl.set_visible(segType in ["Selection"])
        self.selPtsEntry.set_visible(segType in ["Selection"])
        self.maskTypeLbl.set_visible(segType not in ["Auto", "Box"])
        self.maskTypeDropDown.set_visible(segType not in ["Auto", "Box"])
        self.segResLbl.set_visible(isAuto)
        self.segResDropDown.set_visible(isAuto)
        self.cropNLayersLbl.set_visible(isAuto)
        self.cropNLayersChk.set_visible(isAuto)
        self.minMaskAreaLbl.set_visible(isAuto)
        self.minMaskAreaEntry.set_visible(isAuto)

    def on_random_toggled(self, widget):
        is_random = self.randColBtn.get_active()
        self.maskColorLbl.set_visible(not is_random)
        self.maskColorBtn.set_visible(not is_random)

    def get_values(self):
        self.values.pythonPath = self.pythonFileBtn.get_filename()
        self.values.modelType = self.modelTypeVals[self.modelTypeDropDown.get_active()]
        self.values.checkPtPath = self.checkPtFileBtn.get_filename()
        self.values.segType = self.segTypeVals[self.segTypeDropDown.get_active()]
        self.values.maskType = self.maskTypeVals[self.maskTypeDropDown.get_active()]
        if hasattr(self, "randColBtn"):
            self.values.isRandomColor = self.randColBtn.get_active()
            rgba = self.maskColorBtn.get_rgba()
            self.values.maskColor = [
                int(rgba.red * 255),
                int(rgba.green * 255),
                int(rgba.blue * 255),
                255,
            ]
        self.values.selPtCnt = int(self.selPtsEntry.get_text())
        self.values.segRes = self.segResVals[self.segResDropDown.get_active()]
        self.values.cropNLayers = 1 if self.cropNLayersChk.get_active() else 0
        self.values.minMaskArea = int(self.minMaskAreaEntry.get_text())
        # if self.boxPathNameDropDown:
        #    self.values.selBoxPathName = self.boxPathNames[self.boxPathNameDropDown.get_active()]
        self.values.persist(self.configFilePath)
        return self.values


def getPathDict(image):
    return {}


def shellRun(cmdArgs, stdoutFile=None, env_vars=None, useos=False):
    if env_vars is None:
        env_vars = os.environ.copy()

    cmdLine = " ".join(cmdArgs)
    logging.info("Running command: %s" % cmdLine)
    if useos:
        os.system(cmdLine)
    else:
        process = subprocess.Popen(
            cmdArgs,
            env=env_vars,
            stdout=subprocess.PIPE if not stdoutFile else stdoutFile,
            stderr=subprocess.PIPE,
        )
        try:
            # Processing stdout
            if process.stdout:
                for line in iter(process.stdout.readline, b""):
                    # Decode the line if necessary (Python 3 reads bytes from PIPE)
                    line = line.decode("utf-8")
                    print(
                        line,
                    )
            process.wait()

            if process.returncode != 0:
                error_message = "Command failed with the following error:\n "
                error_lines = [
                    line.decode("utf-8") for line in iter(process.stderr.readline, b"")
                ]
                logging.error(error_message + "".join(error_lines))
                return False

        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
    return True


def unpackBoolArray(filepath):
    with open(filepath, "rb") as file:
        packed_data = bytearray(file.read())

    byte_index = 8  # Skip the first 8 bytes for num_rows and num_cols

    num_rows = struct.unpack(">I", packed_data[:4])[0]
    num_cols = struct.unpack(">I", packed_data[4:8])[0]

    unpacked_data = []
    bit_position = 0

    for _ in range(num_rows):
        unpacked_row = []
        for _ in range(num_cols):
            if bit_position == 0:
                current_byte = packed_data[byte_index]
                byte_index += 1

            boolean_value = (current_byte >> bit_position) & 1
            unpacked_row.append(boolean_value)
            bit_position += 1

            if bit_position == 8:
                bit_position = 0

        unpacked_data.append(unpacked_row)

    return unpacked_data


def readMaskFile(filepath, formatBinary):
    if formatBinary:
        return unpackBoolArray(filepath)
    else:
        mask = []
        with open(filepath, "r") as f:
            lines = f.readlines()
        for line in lines:
            mask.append([val == "1" for val in line])
        return mask


def exportSelection(image, expfile, exportCnt):
    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-bounds")
    config = procedure.create_config()
    config.set_property("image", image)
    result = procedure.run(config)
    non_empty = result.index(1)
    x1 = result.index(2)
    y1 = result.index(3)
    x2 = result.index(4)
    y2 = result.index(5)

    if not non_empty:
        return

    coords = []
    numPts = (x2 - x1) * (y2 - y1)
    if exportCnt >= numPts:
        selIdxs = range(numPts)
    else:
        selIdxs = random.sample(range(numPts), exportCnt)
    for selIdx in selIdxs:
        x = x1 + selIdx % (x2 - x1)
        y = y1 + int(selIdx / (x2 - x1))

        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-value")
        config = procedure.create_config()
        config.set_property("image", image)
        config.set_property("x", float(x))
        config.set_property("y", float(y))
        result = procedure.run(config)
        value = result.index(1)

        if value > 200:
            coords.append((x, y))
    with open(expfile, "w") as f:
        for co in coords:
            f.write(str(co[0]) + " " + str(co[1]) + "\n")


def getRandomColor(layerCnt):
    uniqueColors = set()
    while len(uniqueColors) < layerCnt:
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        color = (red, green, blue)

        if color not in uniqueColors:
            uniqueColors.add(color)
    return list(uniqueColors)


def createLayers(image, maskFileNoExt, userSelColor, formatBinary, values):
    width, height = image.get_width(), image.get_height()

    idx = 0
    maxLayers = 99999

    parent = Gimp.GroupLayer.new(image)
    parent.set_name(f"Segment Anything - {values.segType}")
    image.insert_layer(parent, None, 0)
    parent.set_opacity(50)

    uniqueColors = getRandomColor(layerCnt=999)

    if image.get_base_type() == Gimp.ImageType.GRAYA_IMAGE:
        layerType = Gimp.ImageType.GRAYA_IMAGE
        userSelColor = [100, 255]
        babl_format = "YA u8"
        pix_size = 2
    else:
        layerType = Gimp.ImageType.RGBA_IMAGE
        babl_format = "RGBA u8"
        pix_size = 4

    while idx < maxLayers:
        filepath = maskFileNoExt + str(idx) + ".seg"
        if exists(filepath):
            print("Creating Layer..", (idx + 1))
            newlayer = Gimp.Layer.new(
                image,
                f"Mask - {values.segType} #{idx + 1}",
                width,
                height,
                layerType,
                100.0,
                Gimp.LayerMode.NORMAL,
            )
            image.insert_layer(newlayer, parent, 0)
            newlayer.set_visible(False)

            buffer = newlayer.get_buffer()
            rect = Gegl.Rectangle.new(0, 0, width, height)

            maskVals = readMaskFile(filepath, formatBinary)
            maskColor = (
                userSelColor
                if userSelColor is not None
                else list(uniqueColors[idx]) + [255]
            )

            mask_color_bytes = bytes(maskColor)
            transparent_pixel = bytes(pix_size)
            row_byte_strings = []
            for row in maskVals:
                row_pixels = []
                for p in row:
                    if p:
                        row_pixels.append(mask_color_bytes)
                    else:
                        row_pixels.append(transparent_pixel)
                row_byte_strings.append(b"".join(row_pixels))
            pixels = b"".join(row_byte_strings)

            buffer.set(rect, babl_format, pixels)

            idx += 1
            newlayer.update(0, 0, width, height)
        else:
            break

    return idx


def cleanup(filepathPrefix):
    for f in glob.glob(filepathPrefix + "*"):
        os.remove(f)


def showError(message):
    dialog = Gtk.MessageDialog(
        None,
        Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        Gtk.MessageType.ERROR,
        Gtk.ButtonsType.OK,
        message,
    )

    dialog.run()
    dialog.destroy()


def validateOptions(image, values):
    if values.segType in {"Selection", "Box"}:
        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-is-empty")
        config = procedure.create_config()
        config.set_property("image", image)
        result = procedure.run(config)
        isSelEmpty = result.index(1)
        if isSelEmpty:
            showError(
                "No Selection! For the Segmentation Types: "
                + "Selection to work you need "
                + "to select an area on the image"
            )
            return False
    return True


def configLogging(level):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_segmentation(image, values):
    configLogging(logging.DEBUG)
    if not validateOptions(image, values):
        return

    boxPathDict = getPathDict(image)

    if values.checkPtPath is None:
        logging.error("Please set the Segment Anything checkpoint path.")
        return

    if values.pythonPath is None:
        logging.warn("Warning: python path is None trying default python executable")
        pythonPath = "python"
    else:
        pythonPath = values.pythonPath

    formatBinary = True
    filePrefix = "__seg__"
    filepathPrefix = os.path.join(tempfile.gettempdir(), filePrefix)
    selFile = filepathPrefix + "sel__.txt"
    maskFileNoExt = filepathPrefix + "mask__"

    segAnyScriptName = "seganybridge.py"

    cleanup(filepathPrefix)

    currDir = os.path.dirname(os.path.realpath(__file__))
    scriptFilepath = os.path.join(currDir, segAnyScriptName)

    ipFilePath = filepathPrefix + next(tempfile._get_candidate_names()) + ".png"

    cmd = [
        pythonPath,
        scriptFilepath,
        values.modelType,
        values.checkPtPath,
        ipFilePath,
        values.segType,
        values.maskType,
        maskFileNoExt,
        str(formatBinary),
    ]

    if values.segType == "Auto":
        cmd.extend([values.segRes, str(values.cropNLayers), str(values.minMaskArea)])

    newImage = image.duplicate()
    visLayer = newImage.merge_visible_layers(Gimp.MergeType.CLIP_TO_IMAGE)

    procedure = Gimp.get_pdb().lookup_procedure("file-png-export")
    config = procedure.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", newImage)

    gfile = Gio.File.new_for_path(ipFilePath)
    config.set_property("file", gfile)
    config.set_property("interlaced", False)
    config.set_property("compression", 9)
    config.set_property("bkgd", False)
    config.set_property("offs", False)
    config.set_property("phys", False)
    config.set_property("time", False)
    config.set_property("save-transparent", True)
    config.set_property("optimize-palette", False)

    config.set_property("include-exif", False)
    config.set_property("include-iptc", False)
    config.set_property("include-xmp", False)
    config.set_property("include-color-profile", False)
    config.set_property("include-thumbnail", False)
    config.set_property("include-comment", False)
    procedure.run(config)

    newImage.delete()

    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-save")
    config = procedure.create_config()
    config.set_property("image", image)
    result = procedure.run(config)
    channel = result.index(1)

    if values.segType in {"Selection"}:
        exportSelection(image, selFile, values.selPtCnt)
        cmd.append(selFile)
    elif values.segType == "Box":
        procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-bounds")
        config = procedure.create_config()
        config.set_property("image", image)
        result = procedure.run(config)
        x1 = result.index(2)
        y1 = result.index(3)
        x2 = result.index(4)
        y2 = result.index(5)
        cmd.append("sel_place_holder")
        cmd.append(",".join(str(co) for co in [x1, y1, x2, y2]))

    procedure = Gimp.get_pdb().lookup_procedure("gimp-selection-none")
    config = procedure.create_config()
    config.set_property("image", image)
    procedure.run(config)
    shellRun(cmd)

    layerMaskColor = None if values.isRandomColor else values.maskColor
    createLayers(image, maskFileNoExt, layerMaskColor, formatBinary, values)
    cleanup(filepathPrefix)

    if channel is not None:
        procedure = Gimp.get_pdb().lookup_procedure("gimp-image-select-item")
        config = procedure.create_config()
        config.set_property("image", image)
        config.set_property("operation", Gimp.ChannelOps.REPLACE)
        config.set_property("item", channel)
        procedure.run(config)

    logging.debug("Finished creating segments!")


class SegAnyPlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["seg-any-gimp3"]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.seg_any_run, None
        )
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Segment Anything Layers")
        procedure.set_attribution("Shrinivas Kulkarni", "Shrinivas Kulkarni", "2024")
        procedure.add_menu_path("<Image>/Image")
        return procedure

    def seg_any_run(self, procedure, run_mode, image, drawables, config, data):
        boxPathDict = getPathDict(image)
        dialog = OptionsDialog(image, boxPathDict)
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            values = dialog.get_values()
            image.undo_group_start()
            run_segmentation(image, values)
            image.undo_group_end()

        dialog.destroy()

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(SegAnyPlugin.__gtype__, sys.argv)
