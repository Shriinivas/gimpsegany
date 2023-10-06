'''
Gimp plugin for integration with Meta Segment Anything

Copyright (C) 2023  Shrinivas Kulkarni

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

from gimpfu import pdb, register, main, CLIP_TO_IMAGE, RGBA_IMAGE
from gimpfu import LAYER_MODE_NORMAL_LEGACY, GRAY, GRAYA_IMAGE
import tempfile, subprocess
from subprocess import PIPE
from os.path import exists
from array import array
import random, os, glob, struct
import gtk
import json


def shellRun(cmdLine, verbose=True):
    if verbose:
        print(cmdLine)
    process = subprocess.Popen(cmdLine, stdout=PIPE, shell=True)
    process.wait()
    o0, o1 = process.communicate()
    return [x.decode() if x is not None else None for x in [o0, o1]]


def unpackBoolArray(filepath):
    with open(filepath, 'rb') as file:
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
    else:  # Ony for testing
        mask = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # print(line)
            mask.append([val == '1' for val in line])
            # print(mask[-1])
        return mask


def exportSelection(image, expfile, exportCnt):
    # maxCos = 50
    exists, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    coords = []
    numPts = (x2 - x1) * (y2 - y1)
    if exportCnt >= numPts:
        selIdxs = range(numPts)
    else:
        selIdxs = random.sample(range(numPts), exportCnt)
    for selIdx in selIdxs:
        x = x1 + selIdx % (x2-x1)
        y = y1 + int(selIdx / (x2-x1))
        value = pdb.gimp_selection_value(image, x, y)
        if value > 200:
            coords.append((x, y))
    with open(expfile, 'w') as f:
        for co in coords:
            f.write(str(co[0]) + ' ' + str(co[1]) + '\n')


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


def createLayers(image, maskFileNoExt, userSelColor, formatBinary):
    width, height = image.width, image.height

    idx = 0
    maxLayers = 99999

    parent = pdb.gimp_layer_group_new(image)
    pdb.gimp_image_insert_layer(image, parent, None, 0)
    pdb.gimp_layer_set_opacity(parent, 50)

    uniqueColors = getRandomColor(layerCnt=999)

    if image.base_type == GRAY:
        layerType = GRAYA_IMAGE
        userSelColor = [100, 255]
    else:
        layerType = RGBA_IMAGE

    while idx < maxLayers:
        filepath = maskFileNoExt + str(idx) + '.seg'
        if exists(filepath):
            print('Creating Layer..', (idx + 1))
            newlayer = pdb.gimp_layer_new(image, width, height,
                                          layerType, 'Segment Auto', 100,
                                          LAYER_MODE_NORMAL_LEGACY)
            pdb.gimp_image_insert_layer(image, newlayer, parent, 0)
            pdb.gimp_item_set_visible(newlayer, False)
            rgn = newlayer.get_pixel_rgn(0, 0, width, height, True, True)
            pixSize = len(rgn[0, 0])

            pixels = array("B", "\x00" * (width * height * pixSize))

            maskVals = readMaskFile(filepath, formatBinary)
            maskColor = userSelColor if userSelColor is not None \
                else list(uniqueColors[idx]) + [255]
            x = 0
            for line in maskVals:
                for y, p in enumerate(line):
                    if p:
                        pos = (y + width * x) * pixSize
                        pixels[pos: pos + pixSize] = \
                            array('B', maskColor)
                x += 1
            idx += 1
            rgn[0:width, 0:height] = pixels.tostring()
            newlayer.flush()
            newlayer.merge_shadow(True)
            newlayer.update(0, 0, width, height)
        else:
            break

    return idx


def cleanup(filepathPrefix):
    for f in glob.glob(filepathPrefix + '*'):
        os.remove(f)


def getPathDict(image):
    paths = image.vectors
    return {path.name: path for path in paths}


def getBoxCos(image, boxPathDict, pathName):
    path = boxPathDict[pathName]
    points = path.strokes[0].points[0]
    ptsCnt = len(points)

    if ptsCnt != 24:
        print('Error: Path is not a box!', ptsCnt)
        return None
    else:
        topLeft = [points[2], points[3]]
        bottomRight = [points[14], points[15]]
        return topLeft + bottomRight


class DialogValue:
    def __init__(self, filepath):
        data = None
        self.pythonPath = None
        self.modelType = 'vit_h'
        self.checkPtPath = None
        self.maskType = 'Multiple'
        self.segType = 'Auto'
        self.isRandomColor = False
        self.maskColor = [255, 0, 0, 255]
        self.selPtCnt = 50
        self.selBoxPathName = None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.pythonPath = data.get('pythonPath', self.pythonPath)
                self.modelType = data.get('modelType', self.modelType)
                self.checkPtPath = data.get('checkPtPath', self.checkPtPath)
                self.maskType = data.get('maskType', self.maskType)
                self.segType = data.get('segType', self.segType)
                self.isRandomColor = data.get('isRandomColor', self.isRandomColor)
                self.maskColor = data.get('maskColor', self.maskColor)
                self.selPtCnt = data.get('selPtCnt', self.selPtCnt)
        except Exception as e:
            print("Exception", e)
            pass

    def persist(self, filepath):
        data = {
            'pythonPath': self.pythonPath,
            'modelType': self.modelType,
            'checkPtPath': self.checkPtPath,
            'maskType': self.maskType,
            'segType': self.segType,
            'isRandomColor': self.isRandomColor,
            'maskColor': self.maskColor,
            'selPtCnt': self.selPtCnt,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)


def showError(message):
    dialog = gtk.MessageDialog(
        None,
        gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
        gtk.MESSAGE_ERROR,
        gtk.BUTTONS_OK,
        message
    )

    dialog.run()
    dialog.destroy()


# Define a function to restrict input to integers
def kepPressNum(widget, event):
    allowedKeys = set([gtk.keysyms.Home, gtk.keysyms.End, gtk.keysyms.Left,
                       gtk.keysyms.Right, gtk.keysyms.Delete,
                       gtk.keysyms.BackSpace])
    keyval = event.keyval
    # Check if the key is not a digit (0-9) and backspace and delete
    if (keyval < 48 or keyval > 57) and keyval not in allowedKeys:
        return True  # Ignore the keypress
    return False  # Allow the keypress


def onRandomToggled(checkbox, controlsToHide):
    checked = checkbox.get_active()
    for i, control in enumerate(controlsToHide):
        control.set_property('visible', not checked)


def onSegTypeChanged(dropdown, segTypeVals, controlsToHide, maskTypeControls):
    segType = segTypeVals[dropdown.get_active()]
    hideIdxs = []
    if segType == 'Selection':  # Hide Path Names dropdown
        hideIdxs = [1]
    elif segType in {'Auto', 'Box'}:  # Hide Sel Count and Path Names dropdown
        hideIdxs = [0, 1]

    for i in range(len(controlsToHide)):
        for control in controlsToHide[i]:
            if control is not None:
                control.set_property('visible', not (i in hideIdxs))
    for control in maskTypeControls:
        control.set_property('visible', segType != 'Auto')


def getRightAlignLabel(labelStr):
    label = gtk.Label(labelStr)
    alignment = gtk.Alignment(xalign=1, yalign=0.5, xscale=0, yscale=0)
    alignment.add(label)
    return alignment


def validateOptions(image, values):
    if values.segType in {'Selection', 'Box-Selection', 'Box'}:
        isSelEmpty = pdb.gimp_selection_is_empty(image)
        if isSelEmpty == 1:
            showError('No Selection! For the Segmentation Types: Box, ' +
                      'Box-Selection and Selection to work you need ' +
                      'to select an area on the image')
            return False
    return True


def optionsDialog(image, boxPathDict):
    boxPathNames = sorted(boxPathDict.keys())
    boxPathExist = len(boxPathNames) > 0
    isGrayScale = image.base_type == GRAY
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    configFilePath = os.path.join(scriptDir, 'segany_settings.json')

    dialog = gtk.Dialog('Segment Anything', None, gtk.DIALOG_MODAL)
    values = DialogValue(configFilePath)

    pythonPath = values.pythonPath
    print(pythonPath)
    modelType = values.modelType
    checkPtPath = values.checkPtPath
    maskType = values.maskType
    segType = values.segType
    isRandomColor = values.isRandomColor
    maskColor = values.maskColor
    selPtCnt = values.selPtCnt

    modelTypeVals = ['vit_h', 'vit_l', 'vit_b']
    modelTypeIdx = modelTypeVals.index(modelType)

    segTypeVals = ['Auto', 'Box', 'Selection', 'Box-Selection']
    segTypeIdx = segTypeVals.index(segType)

    maskTypeVals = ['Multiple', 'Single']
    maskTypeIdx = maskTypeVals.index(maskType)

    pythonFileLbl = getRightAlignLabel('Python3 Path:')
    pythonFileBtn = gtk.FileChooserButton('Select File')
    if pythonPath is not None:
        pythonFileBtn.set_filename(pythonPath)

    modelTypeLbl = getRightAlignLabel('Checkpoint Type:')
    modelTypeDropDown = gtk.combo_box_new_text()
    for value in modelTypeVals:
        modelTypeDropDown.append_text(value)
    modelTypeDropDown.set_active(modelTypeIdx)

    checkPtFileLbl = getRightAlignLabel('Checkpoint Path:')
    checkPtFileBtn = gtk.FileChooserButton('Select File')
    if checkPtPath is not None:
        checkPtFileBtn.set_filename(checkPtPath)

    maskTypeLbl = getRightAlignLabel('Mask Type:')
    maskTypeDropDown = gtk.combo_box_new_text()
    for value in maskTypeVals:
        maskTypeDropDown.append_text(value)
    maskTypeDropDown.set_active(maskTypeIdx)

    segTypeLbl = getRightAlignLabel('Segmentation Type:')
    segTypeDropDown = gtk.combo_box_new_text()
    for value in segTypeVals:
        segTypeDropDown.append_text(value)
    segTypeDropDown.set_active(segTypeIdx)

    if not isGrayScale:
        maskColorLbl = getRightAlignLabel('Mask Color:')
        colHexVal = '#' + ''.join([('%x' % c).zfill(2) for c in maskColor[:3]])
        gtkColor = gtk.gdk.color_parse(colHexVal)
        maskColorBtn = gtk.ColorButton(gtkColor)  # Default color: Red

        randColBtn = gtk.CheckButton('Random Mask Color')
        randColBtn.set_active(isRandomColor)
        randColBtn.connect('toggled', onRandomToggled,
                           [maskColorLbl, maskColorBtn])

    selPtsLbl = getRightAlignLabel('Selection Points:')
    selPtsEntry = gtk.Entry()
    selPtsEntry.connect('key-press-event', kepPressNum)
    selPtsEntry.set_text(str(selPtCnt))  # Set a default value

    boxPathNameLbl, boxPathNameDropDown = None, None
    if boxPathExist:
        boxPathNameLbl = getRightAlignLabel('Box Path:')
        boxPathNameDropDown = gtk.combo_box_new_text()
        for value in boxPathNames:
            boxPathNameDropDown.append_text(value)
        boxPathNameDropDown.set_active(0)

    table = gtk.Table(0, 0, False)

    segTypeDropDown.connect('changed', onSegTypeChanged,
                            segTypeVals, [[selPtsLbl, selPtsEntry],
                                          [boxPathNameLbl,
                                           boxPathNameDropDown]],
                                         [maskTypeLbl, maskTypeDropDown])

    rowIdx = 0

    table.attach(pythonFileLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(pythonFileBtn, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    table.attach(modelTypeLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(modelTypeDropDown, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    table.attach(checkPtFileLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(checkPtFileBtn, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    table.attach(segTypeLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(segTypeDropDown, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    table.attach(maskTypeLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(maskTypeDropDown, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    table.attach(selPtsLbl, 0, 1, rowIdx, rowIdx + 1)
    table.attach(selPtsEntry, 1, 2, rowIdx, rowIdx + 1)
    rowIdx += 1

    if boxPathExist:
        table.attach(boxPathNameLbl, 0, 1, rowIdx, rowIdx + 1)
        table.attach(boxPathNameDropDown, 1, 2, rowIdx, rowIdx + 1)
        rowIdx += 1

    if not isGrayScale:
        table.attach(randColBtn, 1, 2, rowIdx, rowIdx + 1)
        rowIdx += 1

        table.attach(maskColorLbl, 0, 1, rowIdx, rowIdx + 1)
        table.attach(maskColorBtn, 1, 2, rowIdx, rowIdx + 1)
        rowIdx += 1

        onRandomToggled(randColBtn, [maskColorLbl, maskColorBtn])

    hbox = gtk.HBox()
    hbox.pack_start(table, False, False, 0)

    dialog.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
    dialog.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)

    dialog.vbox.pack_start(hbox, True, True, 0)

    dialog.show_all()

    onSegTypeChanged(segTypeDropDown, segTypeVals,
                     [[selPtsLbl, selPtsEntry],
                      [boxPathNameLbl, boxPathNameDropDown]],
                     [maskTypeLbl, maskTypeDropDown])

    while True:
        response = dialog.run()

        if response == gtk.RESPONSE_OK:
            values.pythonPath = pythonFileBtn.get_filename()
            values.modelType = modelTypeVals[modelTypeDropDown.get_active()]
            values.checkPtPath = checkPtFileBtn.get_filename()
            values.segType = segTypeVals[segTypeDropDown.get_active()]
            values.maskType = maskTypeVals[maskTypeDropDown.get_active()]
            if not isGrayScale:
                values.isRandomColor = randColBtn.get_active()
                maskColor = maskColorBtn.get_color()
                values.maskColor = [255 * maskColor.red / 65535,
                                    255 * maskColor.green / 65535,
                                    255 * maskColor.blue / 65535, 255]
            values.selPtCnt = int(selPtsEntry.get_text())
            if boxPathExist:
                values.selBoxPathName = \
                    boxPathNames[boxPathNameDropDown.get_active()]

            valid = validateOptions(image, values)
            if not valid:
                continue
            values.persist(configFilePath)
        else:
            values = None
        break

    dialog.destroy()
    return values


def plugin_main(image, layer):
    boxPathDict = getPathDict(image)
    values = optionsDialog(image, boxPathDict)
    if values is None:  # Cancelled
        return

    if values.checkPtPath is None:
        print('Please set the Segment Anything checkpoint path.')
        return

    if values.pythonPath is None:
        print('Warning: python path is None trying default python executable')
        pythonPath = 'python'
    else:
        pythonPath = values.pythonPath

    formatBinary = True
    filePrefix = '__seg__'
    filepathPrefix = os.path.join(tempfile.gettempdir(), filePrefix)
    selFile = filepathPrefix + 'sel__.txt'
    maskFileNoExt = filepathPrefix + 'mask__'

    segAnyScriptName = 'seganybridge.py'

    cleanup(filepathPrefix)

    currDir = os.path.dirname(os.path.realpath(__file__))
    scriptFilepath = os.path.join(currDir, segAnyScriptName)

    ipFilePath = filepathPrefix + next(tempfile._get_candidate_names()) \
        + '.png'

    cmd = '%s %s %s %s %s %s %s %s %r' % (pythonPath, scriptFilepath,
                                          values.modelType, values.checkPtPath,
                                          ipFilePath, values.segType,
                                          values.maskType, maskFileNoExt,
                                          formatBinary)

    newImage = pdb.gimp_image_duplicate(image)
    visLayer = pdb.gimp_image_merge_visible_layers(newImage, CLIP_TO_IMAGE)
    pdb.gimp_file_save(newImage, visLayer, ipFilePath, '?')
    pdb.gimp_image_delete(newImage)

    channel = pdb.gimp_selection_save(image)

    if values.segType in {'Selection', 'Box-Selection'}:
        exportSelection(image, selFile, values.selPtCnt)
        cmd += ' ' + selFile
        if values.segType == 'Box-Selection':
            boxCos = getBoxCos(image, boxPathDict, values.selBoxPathName)
            if boxCos is None:
                return
            cmd += ' ' + ','.join(str(co) for co in boxCos)
    elif values.segType == 'Box':
        exists, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
        cmd += ' sel_place_holder ' + ','.join(str(co)
                                               for co in [x1, y1, x2, y2])

    pdb.gimp_selection_none(image)
    shellRun(cmd)

    layerMaskColor = None if values.isRandomColor else values.maskColor
    createLayers(image, maskFileNoExt, layerMaskColor, formatBinary)
    cleanup(filepathPrefix)

    if channel is not None:
        pdb.gimp_image_select_item(image, 2, channel)

    print('Done!')


register(
        "python_fu_seg_any",
        "Segment Anything Mask Layers",
        "Create Layers With Masks Generated By Segment Anything",
        "Shrinivas Kulkarni",
        "Shrinivas Kulkarni",
        "2023",
        "<Image>/Image/Segment Anything Layers...",
        "RGB*, GRAY*",
        [],
        [],
        plugin_main)


main()
