import time
from typing import Dict, Any
import numpy as np
import torch
import cv2
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from wsl.networks.medinet.utils import box_to_map


def engine(loader: Any, checkpoint: Dict[str, Any],
           batchsize: int, classes: int, is_train: bool):
    overall_loss = []
    overall_class_loss = []
    overall_reg_loss = []
    start = time.time()

    with torch.set_grad_enabled(is_train):
        for iter_num, data in enumerate(loader):
            class_loss, reg_loss = checkpoint['model']([data[0].cuda().float(), data[1].cuda().float()])
            class_loss = class_loss.mean()
            reg_loss = reg_loss.mean()
            loss = class_loss + reg_loss

            if is_train:
                loss.backward()
                checkpoint['optimizer'].step()
                checkpoint['optimizer'].zero_grad()

            overall_class_loss.append(float(class_loss.item()))
            overall_reg_loss.append(float(reg_loss.item()))
            overall_loss.append(float(loss.item()))

            speed = batchsize * iter_num // (time.time() - start)
            print('Epoch:', checkpoint['epoch'], 'Iter:', iter_num,
                  'Class Loss', round(np.mean(overall_class_loss), 3),
                  'Reg Loss', round(np.mean(overall_loss), 3),
                  'Loss', round(np.mean(overall_loss), 3),
                  'Speed:', int(speed), 'img/s', end='\r', flush=True)

        class_loss = np.mean(overall_class_loss)
        reg_loss = np.mean(overall_reg_loss)
        loss = np.mean(overall_loss)
        summary = (f'Epoch Summary- Class Loss:{round(class_loss, 3)}, Reg Loss: {round(reg_loss, 3)}, Loss:{round(loss, 3)}')
        print(summary)
        return class_loss, reg_loss, loss, summary


def engine_boxes(loader: Any, checkpoint: Dict[str, Any]):
    checkpoint['model'].eval()
    df = {'Id': [], 'index': [], 'score': [], 'box': []}
    start = time.time()
    all_preds = []
    all_labels = []
    map_scores = []

    org_size = (1024, 1024)
    new_size = (224, 224)

    threshold = 0.1
    box_labels = []
    box_scores = []

    with torch.set_grad_enabled(False):
        for iter_num, data in enumerate(loader):
            scores, indices, boxes = checkpoint['model'](data[0].cuda().float().unsqueeze(dim=0))
            scores, order = torch.sort(scores)

            scores = scores.tolist()
            indices = indices.tolist()
            indices = [indices[i] for i in order]
            boxes = boxes.tolist()
            boxes = [boxes[i] for i in order]

            df['Id'] += [data[3]] * len(scores)
            df['score'] += scores
            df['index'] += indices
            df['box'] += boxes

            all_labels.append(data[2])
            all_preds.append(max(scores + [0.0]))

            ground_map = box_to_map(loader.df[loader.df.Id == data[3]].box.to_list(), np.zeros(org_size))
            predicted_map = box_to_map(boxes, np.zeros(org_size), scores)
            ground_map = cv2.resize(ground_map, new_size, interpolation=cv2.INTER_NEAREST).clip(0, 1)
            predicted_map = cv2.resize(predicted_map, new_size, interpolation=cv2.INTER_AREA).clip(0, 1)
            if data[2] != 0:
                map_scores.append(average_precision_score(ground_map.flatten(), predicted_map.flatten()))

            if len(boxes) > 1:
                box = boxes[-1]
                h, w = (box[0] + box[2] / 2) * new_size[0] / org_size[0], (box[1] + box[3] / 2) * new_size[1] / org_size[1]
                box = [[max(0, int(h - new_size[0] / 14)), max(0, int(w - new_size[1] / 14)), min(new_size[0], int(h + new_size[0] / 14)), min(new_size[1], int(w + new_size[1] / 14))]]
                score = scores[-1:]
                predicted_map = box_to_map(box, np.zeros(new_size), score)
            else:
                predicted_map = np.zeros(new_size)
            overlap = np.sum(np.multiply(ground_map, predicted_map)) / np.sum(predicted_map)
                
            if data[2] == 1 and overlap < threshold:
                box_labels += [0, 1]
                box_scores += [np.max(predicted_map), 0]
            else:
                box_labels += [data[2]]
                box_scores += [np.max(predicted_map)]

            speed = iter_num // (time.time() - start)
            print('Epoch:', checkpoint['epoch'], 'Iter:', iter_num, 'Score:', np.mean(map_scores), 'Speed:', int(speed), 'img/s', end='\r', flush=True)

    df = pd.DataFrame.from_dict(df)
    rmetric = roc_auc_score(all_labels, all_preds)
    wild_metric = np.mean(map_scores)
    pointwise = average_precision_score(box_labels, box_scores)
    return df, rmetric, wild_metric, pointwise
