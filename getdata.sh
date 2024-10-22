#!/bin/bash
set -e

# Set absolute path
export PYTHONPATH="$PWD/dataset/:$PWD/dataset/data_conala/:$PWD/model/:$PWD/dataset/data_conala/conala-corpus/"
echo "$PYTHONPATH"

# Get the data
echo "download CoNaLa dataset"
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip -d ./dataset/data_conala
rm -r conala-corpus-v1.1.zip
echo "CoNaLa done"

# Preprocess data

python get_data.py \
    config/config.yml