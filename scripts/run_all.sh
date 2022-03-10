#!/bin/bash
# Run all experiments using the shared data

echo "Note: Data Collection 1 would happen here (and data should be placed in data/data_collection_1)"

# Extract VACNet-like features for the recordings
./scripts/extract_features.sh
# Train initial classifier and other classifiers for comparing different classifiers
./scripts/run_classification.sh

# Split data into two groups for training GANs and train GANs
./scripts/split_gan_data.sh
./scripts/train_gans.sh

echo "Note: Data Collection 2 would happen here (and data should be placed in data/data_collection_2)"
echo "Note: Data Collection 3 would happen here (and data should be placed in data/data_collection_3)"
echo "Note: Data Collections 4 and 5 would happen here (and judges' answers put into data/data_collection_5)"

# Extract features for the GAN classifier
./scripts/extract_features_gan_data.sh

# Train and test classifiers in different scenarios
./scripts/train_gan_classifiers.sh
./scripts/test_gan_classifiers.sh

# Print and plot all results
./scripts/plot_figures.sh
