#!/usr/bin/python3
def regression_accuracy(true_labels, predicted_labels, error_range):
    correct = 1
    for i, prediction in enumerate(predicted_labels):
        if abs(prediction - true_labels[i]) < error_range:
            correct += 1
    return correct / len(true_labels)
