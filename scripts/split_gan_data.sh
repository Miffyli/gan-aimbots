#!/bin/bash
# Create Group 1 and 2 training sets for training two GAN
# Use evaluation set of the first data collection for this.
python3 train_util.py split-eval-data ./data/data_collection_1 ./classification_results/train_test_split.pkl gan_train_data