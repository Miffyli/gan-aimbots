#!/bin/bash
# Initial training of DNN and training of different models for comparison (only on light/strong aimbot data)
mkdir -p classification_results
# First train only the DNN part. This will also create train-test split
python3 classification.py features classification_results dnn
# Then train other models if auto-sklearn is present (separated to different script in case you only want to run DNN experiments)
python3 classification.py features classification_results random_forest decision_tree libsvm_svc bernoulli_nb sgd lda
# And plot the results
python3 plot.py classification-results dummy --inputs classification_results/*_scores.npz