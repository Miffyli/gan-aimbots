#!/bin/bash
# Test GAN classifiers in different ways
# Run after train_gan_classifiers.sh

mkdir -p evaluation_scores

# Worst-case scenario: GAN aimbots are completely
# new and not included in the training set.
# Test against the original DNN
python3 classification.py dummy evaluation_scores/worst_case.npz dnn --model-path classification_results/dnn_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl

# Known-attack: Some GAN data is included in the training
# set, but not specifically the one used by the hackers.
# This also includes oracle: Data from the same aimbot
python3 classification.py dummy evaluation_scores/known_attack_group1.npz dnn --model-path gan_classification_results/dnn_group0_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl
python3 classification.py dummy evaluation_scores/known_attack_group2.npz dnn --model-path gan_classification_results/dnn_group1_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl

# Best-case: Training set also includes evaluation data
python3 classification.py dummy evaluation_scores/best_case.npz dnn --model-path gan_classification_results/dnn_all_train_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl

# -- Repeat above on original aimbots for completeness --
python3 classification.py dummy evaluation_scores/trained_on_light.npz dnn --model-path gan_classification_results/dnn_aimbot1_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl
python3 classification.py dummy evaluation_scores/trained_on_strong.npz dnn --model-path gan_classification_results/dnn_aimbot2_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl

# Best-case (training data contains testing data)
python3 classification.py dummy evaluation_scores/best_case_original.npz dnn --model-path gan_classification_results/dnn_original_all_train_model.pkl --feature-files gan_classification_data/* --train-test-split gan_classification_results/train_test_split.pkl