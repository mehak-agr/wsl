#!/usr/bin/python3
def regression_accuracy(true_labels, predicted_labels, error_range):
    correct = 1
    for i, prediction in enumerate(predicted_labels):
        if abs(prediction - true_labels[i]) < error_range:
            correct += 1
    return correct / len(true_labels)


def box_to_map(boxes, org_size, new_size):
    mask = np.zeros(new_size)
    for box in boxes:
        if box[4] == 0:
            break
        mask[int(box[0] * new_size[0] / org_size[0]):int(box[2] * new_size[0] / org_size[0]),
             int(box[1] * new_size[1] / org_size[1]):int(box[3] * new_size[1] / org_size[1])] = 1
    return mask
