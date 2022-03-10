#!/bin/bash
mkdir -p features
for filename in data/data_collection_1/*; do
    python3 feature_extraction.py vacnet $filename features/$(basename $filename)
done

