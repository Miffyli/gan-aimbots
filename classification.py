#!/usr/bin/env python3
#
# classification.py
# Read extracted features, train and evaluate
# different ML algorithms for classifying cheats from legit players
#
import argparse
import glob
import random
import pickle
import os
import math

import numpy as np

AVAILABLE_CLASSIFIERS = [
    "dnn",
    "random_forest",
    "decision_tree",
    "libsvm_svc",
    "bernoulli_nb",
    "sgd",
    "lda"
]

parser = argparse.ArgumentParser("Do classification on extracted features")
parser.add_argument("features_dir", help="Directory/file where feature npz files are stored (from feature_extraction.py)")
parser.add_argument("output_path", help="Path where classification outputs are stored (splits, trained models, scores)")
parser.add_argument("classifiers", type=str, nargs="+", choices=AVAILABLE_CLASSIFIERS, help="Classifiers to run data through")
parser.add_argument("--autosklearn-time-limit", type=int, default=5 * 60 * 60, help="Amount of time given per auto-sklearn run")
parser.add_argument("--eval-ratio", type=float, default=0.2, help="Amount of data to keep for evaluation")
parser.add_argument("--model-path", default=None, type=str, help="If given, use this model to evaluate given feature files.")
parser.add_argument("--train-test-split", default="train_test_split.pkl", type=str, help="File where train-test split is stored")
parser.add_argument("--included-aimbots", default=[1, 2], type=int, nargs="+", help="Aimbot IDs that are included in training/testing.")
parser.add_argument("--model-postfix", type=str, default="", help="String to append to models.")
parser.add_argument("--feature-files", type=str, nargs="*", default=None, help="Direct path to feature files that should be processed.")

NORMALIZATION_FILE_NAME = "feature_normalization.npz"

AUTO_SKLEARN_N_JOBS = 4

DNN_BATCH_SIZE = 64
DNN_NUM_EPOCHS = 50
DNN_VALIDATION_RATIO = 0.1


def train_dnn(args, X_train, y_train, model_path):
    import torch
    from torch.nn import functional as F
    from collections import deque

    from tqdm import tqdm

    # To balance the training.
    class_weights = [
        y_train.mean(),
        (1 - y_train).mean()
    ]

    num_validation = int(X_train.shape[0] * DNN_VALIDATION_RATIO)
    idxs = np.arange(X_train.shape[0])
    np.random.shuffle(idxs)
    validation_idxs = idxs[:num_validation]
    train_idxs = idxs[num_validation:]
    X_valid = X_train[validation_idxs]
    y_valid = y_train[validation_idxs]
    X_train = X_train[train_idxs]
    y_train = y_train[train_idxs]

    class_weights = torch.from_numpy(np.array(class_weights)).float().cuda()

    X_train = torch.from_numpy(X_train).float().cuda()
    y_train = torch.from_numpy(y_train).long().cuda()
    X_valid = torch.from_numpy(X_valid).float().cuda()
    y_valid = torch.from_numpy(y_valid).long().cuda()

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2)
    ).cuda()

    num_iters = (X_train.shape[0] // DNN_BATCH_SIZE) * DNN_NUM_EPOCHS

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    all_indeces = np.arange(X_train.shape[0])
    np.random.shuffle(all_indeces)
    cur_epoch_idx = 0

    losses = deque(maxlen=1000)
    for i in tqdm(range(num_iters)):
        random_idxs = all_indeces[cur_epoch_idx:cur_epoch_idx + DNN_BATCH_SIZE]
        cur_epoch_idx += DNN_BATCH_SIZE
        if cur_epoch_idx > (X_train.shape[0] - DNN_BATCH_SIZE):
            np.random.shuffle(all_indeces)
            cur_epoch_idx = 0

        inputs = X_train[random_idxs]
        targets = y_train[random_idxs]

        predictions = model(inputs)

        loss_output = loss(predictions, targets)

        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        losses.append(loss_output.item())

        if (i % 1000) == 0:
            valid_loss = None
            with torch.no_grad():
                valid_loss = loss(model(X_valid), y_valid).item()
            tqdm.write("Train Loss: {:.5f}  Valid loss: {:.5f}".format(
                sum(losses) / len(losses),
                valid_loss
            ))

    model = model.cpu()
    torch.save(model, model_path)

    # Will be assigned to the class below
    def predict_scores(x):
        x = torch.from_numpy(x).float()
        scores = model(x)
        return scores.cpu().detach().numpy()

    return predict_scores


def train_auto_sklearn(classifier, args, X_train, y_train, model_path):
    import autosklearn.classification
    model = autosklearn.classification.AutoSklearnClassifier(
        include_preprocessors=["no_preprocessing", ],
        include_estimators=[classifier, ],
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        time_left_for_this_task=args.autosklearn_time_limit,
        n_jobs=AUTO_SKLEARN_N_JOBS,
        ml_memory_limit=32000
    )

    model.fit(X_train.copy(), y_train.copy())

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    def predict_scores(x):
        """
        Not really a score...
        """
        return model.predict(x)

    return predict_scores


def load_or_create_split(ids, eval_ratio, file_path):
    """
    Check if file exists for train/test split.
    If not, create a split of ids where
    at least eval_ratio of ids are in evaluation.

    Returns (training_ids, testing_ids)
    """
    testing_ids = None
    training_ids = None
    if os.path.isfile(file_path):
        # Load from disk
        with open(file_path, "rb") as f:
            print("Loading data split from {}".format(file_path))
            split_data = pickle.load(f)
            training_ids = split_data["training_ids"]
            testing_ids = split_data["testing_ids"]
    else:
        # Make a split
        # Select players (computers) to keep
        # for training and testing
        print("Creating split")
        ids = set(ids)
        testing_ids = set(random.sample(ids, math.ceil(len(ids) * args.eval_ratio)))
        training_ids = ids - testing_ids

        # Save for later use
        with open(os.path.join(args.output_path, "train_test_split.pkl"), "wb") as f:
            pickle.dump(
                {"training_ids": training_ids, "testing_ids": testing_ids},
                f
            )
    return training_ids, testing_ids


def get_player_id(filename):
    """
    Return id of the player (hardware id + timestamp)
    for a given path to features or recordings
    """
    filename_split = os.path.basename(filename).split("_")
    # Timestamp + hardware id
    unique_id = filename_split[0] + "_" + filename_split[1]
    return unique_id


def main_evaluate(args):
    import torch

    if len(args.classifiers) > 1 or args.classifiers[0] != "dnn":
        raise ValueError("Only DNN is supported for evaluation")
    if not os.path.isfile(args.features_dir) and args.feature_files is None:
        raise ValueError("features_dir should be path to a feature file, or feature-files should be list of such.")

    # Load normalization data
    normalization_stats = np.load(os.path.join(os.path.dirname(args.model_path), NORMALIZATION_FILE_NAME))
    model = torch.load(args.model_path)

    # Try loading training-testing split. If could not load, just skip
    testing_ids = None
    if os.path.isfile(args.train_test_split):
        print("Loading testing IDs from {}".format(args.train_test_split))
        split = pickle.load(open(args.train_test_split, "rb"))
        testing_ids = split["testing_ids"]
    else:
        print("WARNING: Could not find train-test split file {}. Evaluating all given features!".format(args.train_test_split))

    feature_files = []
    if args.feature_files is None:
        feature_files = [args.features_dir]
    else:
        feature_files = args.feature_files

    features = []
    labels = []
    aimbot_classes = []

    for filename in feature_files:
        if testing_ids is not None:
            player_id = get_player_id(filename)
            if player_id not in testing_ids:
                continue
        # Features_dir is path to a npz file
        data = np.load(filename)

        features.append(data["features"])
        labels.append(data["labels"])
        aimbot_classes.append(data["aimbot_class"])
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    aimbot_classes = np.concatenate(aimbot_classes, axis=0)

    normalized_features = (features - normalization_stats["means"]) / normalization_stats["stds"]

    scores = model(torch.from_numpy(normalized_features).float()).detach().numpy()

    # Use same naming as with main code
    np.savez(
        args.output_path,
        test_features=features,
        test_scores=scores,
        test_labels=labels,
        test_aimbots=aimbot_classes
    )


def main(args):
    data_files = []
    if args.feature_files is None:
        data_files = glob.glob(os.path.join(args.features_dir, "*"))
    else:
        data_files = args.feature_files
    # NOTE:
    # IDs attempt to be unique to each player.
    # They are a concatenation of timestamp when
    # files were sent along with the semi-unique hardware id.
    # This works because data uploading script
    # fixed these two when receiving files.
    ids = []
    datas = []
    for data_file in data_files:
        data = np.load(data_file)
        datas.append(data)
        unique_id = get_player_id(data_file)
        ids.append(unique_id)

    split_path = os.path.join(args.output_path, args.train_test_split)
    training_ids, testing_ids = load_or_create_split(set(ids), args.eval_ratio, split_path)

    print("Using {} hardware-ids for training and {} for evaluation".format(len(training_ids), len(testing_ids)))

    # Load and split the data
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    aimbots_test = []
    for data, unique_id in zip(datas, ids):
        features = data["features"]
        labels = data["labels"]
        # All data per file uses same aimbot index
        aimbots = data["aimbot_class"]
        aimbot = aimbots[0]
        if aimbot in args.included_aimbots or aimbot == 0:
            if unique_id in training_ids:
                X_train.append(features)
                y_train.append(labels)
            else:
                X_test.append(features)
                y_test.append(labels)
                aimbots_test.append(data["aimbot_class"])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    aimbots_test = np.concatenate(aimbots_test, axis=0)

    # Get normalization parameters from training set or from file
    # if it exists.
    features_mean = None
    features_std = None
    normalization_file_path = os.path.join(args.output_path, NORMALIZATION_FILE_NAME)
    if os.path.isfile(normalization_file_path):
        print("Loading normalization stats from {}".format(NORMALIZATION_FILE_NAME))
        normalization_data = np.load(normalization_file_path)
        features_mean = normalization_data["means"]
        features_std = normalization_data["stds"]
    else:
        # Do not normalize last item in features (binary)
        features_mean = np.append(X_train[:, :-1].mean(axis=0), 0)
        features_std = np.append(X_train[:, :-1].std(axis=0), 1)
        np.savez(normalization_file_path, means=features_mean, stds=features_std)

    # Normalize features
    X_train = (X_train - features_mean) / features_std
    X_test = (X_test - features_mean) / features_std

    # TODO run through different classifiers. Create, train,
    # evaluate and store scores.
    for classifier_name in args.classifiers:
        predict_scores = None
        model_path = os.path.join(args.output_path, "{}{}_model.pkl".format(classifier_name, args.model_postfix))
        if classifier_name == "dnn":
            predict_scores = train_dnn(args, X_train, y_train, model_path)
        else:
            predict_scores = train_auto_sklearn(classifier_name, args, X_train, y_train, model_path)

        # Do predictions
        train_scores = predict_scores(X_train)
        test_scores = predict_scores(X_test)

        # Save scores
        output_path = os.path.join(args.output_path, "{}{}_scores.npz".format(classifier_name, args.model_postfix))
        np.savez(
            output_path,
            train_labels=y_train,
            train_scores=train_scores,
            test_labels=y_test,
            test_scores=test_scores,
            test_aimbots=aimbots_test
        )


if __name__ == '__main__':
    args = parser.parse_args()
    if args.model_path:
        main_evaluate(args)
    else:
        main(args)
