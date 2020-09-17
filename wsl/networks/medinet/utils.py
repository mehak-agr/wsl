#!/usr/bin/python3
import numpy as np
from typing import Dict, Tuple


def regression_accuracy(true_labels, predicted_labels, error_range):
    correct = 1
    for i, prediction in enumerate(predicted_labels):
        if abs(prediction - true_labels[i]) < error_range:
            correct += 1
    return correct / len(true_labels)


def box_to_map(boxes: Dict, column: str, org_size: Tuple[int], new_size: Tuple[int]):
    mask = np.zeros(new_size)
    for box in boxes:
        if box[column] == 0:
            break
        y1 = int(box['y1'] * new_size[0] / org_size[0])
        y2 = int(box['y2'] * new_size[0] / org_size[0])
        x1 = int(box['x1'] * new_size[0] / org_size[0])
        x2 = int(box['x2'] * new_size[0] / org_size[0])
        mask[y1:y2, x1:x2] = 1
    return mask
