#!/usr/bin/env bash


. ./cmd.sh
. ./path.sh


set -e -o pipefail -u


stage=3

# Download data
if [ $stage -le 0 ]; then
	mkdir db
        ./local/download_data.sh
fi


# Create wav.scp files for each language
if [ $stage -le 1 ]; then
        ./local/create_scp.sh
fi


# Make MFCC features for each language
if [ $stage -le 2 ]; then
        for folder in db/cu-multilang-dataset/*; do
          echo "making mfcc + pitch features for $folder" 
          ./local/make_mfcc_pitch.sh --nj 30  $folder
	  ./local/compute_cmvn_stats.sh $folder
        done
fi

# Make MFCC features for each language
if [ $stage -le 3 ]; then
	python3 train_model.py 4 1 15 0.8
fi
