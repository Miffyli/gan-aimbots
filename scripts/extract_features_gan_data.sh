#!/bin/bash
mkdir -p gan_features
for filename in data/data_collection_2/*; do
    python3 feature_extraction.py vacnet $filename gan_features/$(basename $filename)
done
