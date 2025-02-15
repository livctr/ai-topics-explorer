#!/bin/bash

DATA_DIR="~/data/arxiv_data"
PATTERN="arxiv-*.json"

# Check if any matching files exist
if ls "$DATA_DIR"/$PATTERN 1> /dev/null 2>&1; then
    echo "Dataset already exists. Skipping download."
else
    echo "No dataset found. Downloading..."
    kaggle datasets download -f arxiv-metadata-oai-snapshot.json -p "$DATA_DIR" --unzip "Cornell-University/arxiv"
fi

