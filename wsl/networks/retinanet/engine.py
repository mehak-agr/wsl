import time
from typing import Dict, Any
import numpy as np
import torch


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
