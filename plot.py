#
# plot.py
# Visualizing data from different scripts
#
import os
from argparse import ArgumentParser

import numpy as np


def standardize_score_format(scores):
    """
    Get a ndarray of scores and
    convert them into (N, 2)
    scores, where first is "likelihood"
    for bona fide player and second
    is for cheating player.
    """
    if scores.ndim == 1:
        # Assume we have binary classification.
        # Turn this into 0.0 / 1.0 scores
        one_hot = np.eye(2)
        scores = one_hot[scores]
    return scores


def plot_classification_results(unparsed_args):
    parser = ArgumentParser("Go through classification score files and report results (see classification.py)")
    parser.add_argument("--inputs", type=str, required=True, nargs="+", help="Score .npz files to plot")
    parser.add_argument("output", type=str, help="Where to store store plots")
    args = parser.parse_args(unparsed_args)

    names = []
    datas = []
    for input_path in args.inputs:
        names.append(os.path.basename(input_path))
        data = np.load(input_path)
        data = dict(**data)
        # Standardize the score type, since some
        # classifiers only give fixed scores
        data["train_scores"] = standardize_score_format(data["train_scores"])
        data["test_scores"] = standardize_score_format(data["test_scores"])
        datas.append(data)

    # Print accuracies
    print("---Accuracies (balanced, argmax)---")
    print("{:<30} {:<10} {:<10}".format("Name", "Train", "Test"))
    for name, data in zip(names, datas):
        train_prediction = np.argmax(data["train_scores"], axis=1)
        train_labels = data["train_labels"]
        train_accuracy = (
            (1 - train_labels.mean()) * train_prediction[train_labels == 1].mean() +
            train_labels.mean() * (1 - train_prediction[train_labels == 0]).mean()
        )
        test_prediction = np.argmax(data["test_scores"], axis=1)
        test_labels = data["test_labels"]
        test_accuracy = (
            (1 - test_labels.mean()) * test_prediction[test_labels == 1].mean() +
            test_labels.mean() * (1 - test_prediction[test_labels == 0]).mean()
        )

        print("{:<30} {:<10.2f} {:<10.2f}".format(name, train_accuracy * 100, test_accuracy * 100))


AVAILABLE_OPERATIONS = {
    "classification-results": plot_classification_results,
}


if __name__ == '__main__':
    parser = ArgumentParser("Different plot utils")
    parser.add_argument("operation", choices=list(AVAILABLE_OPERATIONS.keys()), help="Operation to run")
    args, unparsed_args = parser.parse_known_args()

    operation_fn = AVAILABLE_OPERATIONS[args.operation]
    operation_fn(unparsed_args)
