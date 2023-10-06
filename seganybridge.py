'''
Script to generate Meta Segment Anything masks.

Adapted from:
https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

Author: Shrinivas Kulkarni

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

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, \
    SamAutomaticMaskGenerator, SamPredictor
import sys


def packBoolArray(filepath, arr):
    packed_data = bytearray()
    num_rows = len(arr)
    num_cols = len(arr[0])

    # Add num_rows and num_cols as 32-bit integers at the beginning
    packed_data.extend([num_rows >> 24, (num_rows >> 16) & 255, (num_rows >> 8)
                        & 255, num_rows & 255])
    packed_data.extend([num_cols >> 24, (num_cols >> 16) & 255, (num_cols >> 8)
                        & 255, num_cols & 255])

    current_byte = 0
    bit_position = 0

    for row in arr:
        for boolean_value in row:
            if boolean_value:
                current_byte |= (1 << bit_position)
            bit_position += 1

            if bit_position == 8:
                packed_data.append(current_byte)
                current_byte = 0
                bit_position = 0

    if bit_position > 0:
        packed_data.append(current_byte)

    with open(filepath, 'wb') as f:
        f.write(packed_data)

    return packed_data


def saveMask(filepath, maskArr, formatBinary):
    if formatBinary:
        packBoolArray(filepath, maskArr)
    else:
        with open(filepath, 'w') as f:
            for row in maskArr:
                f.write(''.join(str(int(val)) for val in row) + '\n')


def saveMasks(masks, saveFileNoExt, formatBinary):
    for i, mask in enumerate(masks):
        filepath = saveFileNoExt + str(i) + '.seg'
        arr = [[val for val in row] for row in mask]
        saveMask(filepath, arr, formatBinary)


def segmentAuto(sam, cvImage, saveFileNoExt, formatBinary):
    height, width = cvImage.shape[:2]
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(cvImage)
    masks = [mask['segmentation'] for mask in masks]
    saveMasks(masks, saveFileNoExt, formatBinary)


def segmentBox(sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary):
    predictor = SamPredictor(sam)
    predictor.set_image(cvImage)

    print('boxCos--->', boxCos)
    input_box = np.array(boxCos)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=(maskType == 'Multiple'),
    )
    saveMasks(masks, saveFileNoExt, formatBinary)


def segmentSel(sam, cvImage, maskType, selFile, boxCos, saveFileNoExt,
               formatBinary):
    height, width = cvImage.shape[:2]
    pts = []
    with open(selFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cos = line.split(' ')
            pts.append([int(cos[0]), int(cos[1])])

    predictor = SamPredictor(sam)
    predictor.set_image(cvImage)

    input_point = np.array(pts)
    input_label = np.array([1 for i in range(len(input_point))])

    if boxCos is None:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=(maskType == 'Multiple'),
        )
    else:
        input_box = np.array(boxCos)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == 'Multiple'),
        )
    saveMasks(masks, saveFileNoExt, formatBinary)


def runTest(sam):
    npArr = np.zeros((50, 50), np.uint8)
    cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)

    predictor = SamPredictor(sam)
    predictor.set_image(cvImage)

    input_box = np.array([10, 10, 20, 20])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )


# python seganybridge.py vit_h /path/to/checkpt /ip/file/path Auto Multiple /tmp/__seg__ True /sel/file/path boxCos
# Test command: python seganybridge.py vit_h /path/to/checkpt
def main():
    selFile = None
    modelType = sys.argv[1]
    checkPtFilePath = sys.argv[2]
    sam = sam_model_registry[modelType](checkpoint=checkPtFilePath)
    if torch.cuda.is_available():
        sam.to(device='cuda')
    if len(sys.argv) == 3:
        runTest(sam)
        print('Success!!')
        return

    ipFile = sys.argv[3]
    segType = sys.argv[4]  # auto, selection
    maskType = sys.argv[5]  # Multiple, Single
    saveFileNoExt = sys.argv[6]
    formatBinary = True

    if len(sys.argv) > 7:
        formatBinary = sys.argv[7] == 'True'

    if len(sys.argv) > 8:
        selFile = sys.argv[8]
    elif segType == 'selection':
        print('Please specify selection file path as the last argument')
        assert False

    boxCos = None
    if len(sys.argv) > 9:
        boxCos = [float(val.strip()) for val in sys.argv[9].split(',')]
    elif segType in {'Box', 'Box-Selection'}:
        print('Please specify box top left and bottom-right coordinates')
        assert False

    cvImage = cv2.imread(ipFile)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)

    if segType == 'Auto':
        segmentAuto(sam, cvImage, saveFileNoExt, formatBinary)
    elif segType in {'Selection', 'Box-Selection'}:
        segmentSel(sam, cvImage, maskType, selFile, boxCos, saveFileNoExt,
                   formatBinary)
    elif segType == 'Box':
        segmentBox(sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary)
    else:
        assert False

    print('Done!')


main()
