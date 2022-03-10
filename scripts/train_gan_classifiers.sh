#!/bin/bash
# Create new training/testing sets given the new GAN data,
# move data to new folder and train new classifiers
mkdir -p gan_classification_data gan_classification_results

# Update train-test split with GAN data
# A sensible split (fifty-fifty, with more for training set)
python3 ./train_util.py update-split ./gan_features/ ./classification_results/train_test_split.pkl gan_classification_results/train_test_split.pkl
# A best-case split: all data goes to training (this will be used to train best-case classifier)
python3 ./train_util.py update-split ./gan_features/ ./classification_results/train_test_split.pkl gan_classification_results/train_all_gan_split.pkl --eval-ratio 0.0
# A best-case split for original aimbots too: train on all data
python3 ./train_util.py update-split ./features/ none gan_classification_results/original_train_all_split.pkl --eval-ratio 0.0
# Copython3 normalization stats too
cp ./classification_results/feature_normalization.npz ./gan_classification_results

# Copython3 original and new data into a single folder (to keep the original data untouched)
cp ./features/* ./gan_features/* ./gan_classification_data

# Train two models, both trained on different GAN-aimbots.
# aimbot id 10 is for group 0 GAN, 11 for group 1.
for group_id in 0 1; do
    # Train on proper train-test split
    python3 classification.py gan_classification_data gan_classification_results dnn --model-postfix _group${group_id} --included-aimbots 1${group_id}
done

# Train on all GAN data (best-case scenario)
python3 classification.py gan_classification_data gan_classification_results dnn --model-postfix _all_train --included-aimbots 10 11 --train-test-split train_all_gan_split.pkl

# For completeness, do same as above for strong and light aimbots (train on one, test on another)
for aimbot in 1 2; do
    # Train on proper train-test split
    python3 classification.py gan_classification_data gan_classification_results dnn --model-postfix _aimbot${aimbot} --included-aimbots ${aimbot}
done
# Train on all heuristic data (best-case for strong/light aimbot)
python3 classification.py gan_classification_data gan_classification_results dnn --model-postfix _original_all_train --included-aimbots 1 2 --train-test-split original_train_all_split.pkl
