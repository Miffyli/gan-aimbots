#
# train_util.py
# Different utilities for training data and splitting
#
import os
import pickle
import glob
import random
import math
from argparse import ArgumentParser
import shutil


def split_eval_data(unparsed_args):
    """
    Take train_test_split file from classification.py,
    split the eval set into different groups and copy
    files under a new directory.

    Used to create different training sets for GANs.
    """
    parser = ArgumentParser("Split recording files of eval-set players into different groups")
    parser.add_argument("recording_dir", type=str, help="Directory where recordings reside.")
    parser.add_argument("train_eval_split", type=str, help="Path to train_eval split file from classification.py.")
    parser.add_argument("output", type=str, help="Directory where different groups are created.")
    parser.add_argument("--num-sets", type=int, default=2, help="Number of groups to create.")
    parser.add_argument("--num-remove-ids", type=int, default=1, help="Number of ids to remove before split.")
    args = parser.parse_args(unparsed_args)

    split = pickle.load(open(args.train_eval_split, "rb"))
    eval_ids = list(split["testing_ids"])
    random.shuffle(eval_ids)
    # Remove any ids if we want to do so
    eval_ids = eval_ids[:-args.num_remove_ids]

    recording_files = glob.glob(os.path.join(args.recording_dir, "*"))

    # Split evaluation IDs to groups of --num-sets
    items_per_group = len(eval_ids) / args.num_sets
    assert int(items_per_group) == items_per_group, "Could not make an even split with {} eval ids".format(len(eval_ids))
    items_per_group = int(items_per_group)

    random.shuffle(eval_ids)

    eval_groups = [eval_ids[i:i + items_per_group] for i in range(0, len(eval_ids), items_per_group)]

    # Go over groups, create output directories
    # and put recordings there
    for i, eval_group in enumerate(eval_groups):
        output_dir = os.path.join(args.output, "group{}".format(i))
        os.makedirs(output_dir)
        for recording_file_path in recording_files:
            for eval_id in eval_group:
                if eval_id in recording_file_path:
                    destination_file = os.path.join(output_dir, os.path.basename(recording_file_path))
                    shutil.copy(recording_file_path, destination_file)


def update_train_test_split_with_gan(unparsed_args):
    """
    Take existing train-test split file, and bunch of GAN-aimbot features.
    Split the new data into train-test split and append them into the
    train-test split file.
    """
    parser = ArgumentParser("Update train-test split with GAN-aimbot data")
    parser.add_argument("features_dir", type=str, help="Directory where features reside.")
    parser.add_argument("train_eval_split", type=str, help="Path to train_eval split file from classification.py.")
    parser.add_argument("output", type=str, help="Path where updated train-test split should be stored.")
    parser.add_argument("--eval-ratio", type=float, default=0.45, help="Amount of data to keep for evaluation")
    args = parser.parse_args(unparsed_args)

    data_files = glob.glob(os.path.join(args.features_dir, "*"))
    ids = []
    for data_file in data_files:
        filename_split = os.path.basename(data_file).split("_")
        # Timestamp + hardware id
        unique_id = filename_split[0] + "_" + filename_split[1]
        ids.append(unique_id)

    ids = set(ids)
    testing_ids = set(random.sample(ids, math.ceil(len(ids) * args.eval_ratio)))
    training_ids = ids - testing_ids

    if args.train_eval_split != "none":
        original_split = pickle.load(open(args.train_eval_split, "rb"))
    else:
        original_split = {"training_ids": set(), "testing_ids": set()}

    original_split["training_ids"].update(training_ids)
    original_split["testing_ids"].update(testing_ids)

    with open(args.output, "wb") as f:
        pickle.dump(original_split, f)


AVAILABLE_OPERATIONS = {
    "split-eval-data": split_eval_data,
    "update-split": update_train_test_split_with_gan,
}

if __name__ == '__main__':
    parser = ArgumentParser("Different utils for training")
    parser.add_argument("operation", choices=list(AVAILABLE_OPERATIONS.keys()), help="Operation to run")
    args, unparsed_args = parser.parse_known_args()

    operation_fn = AVAILABLE_OPERATIONS[args.operation]
    operation_fn(unparsed_args)
