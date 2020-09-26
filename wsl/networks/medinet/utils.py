#!/usr/bin/python3
import numpy as np
from typing import Dict, Tuple, List


def regression_accuracy(true_labels, predicted_labels, error_range):
    correct = 1
    for i, prediction in enumerate(predicted_labels):
        if abs(prediction - true_labels[i]) < error_range:
            correct += 1
    return correct / len(true_labels)


def box_to_map(boxes: List, mask):
    # box = [x1, y1, x2, y2]
    for box in boxes:
        mask[box[1]:box[3], box[0]:box[2]] = 1
    return mask


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)


def rle2mask(rle, mask):
    width, height = mask.shape
    mask = mask.flatten
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)
