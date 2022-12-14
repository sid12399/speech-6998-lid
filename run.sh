#!/usr/bin/env bash


. ./cmd.sh
. ./path.sh


set -e -o pipefail -u


stage=0

# Download data
mkdir db
if [ $stage -le 0 ]; then
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
          ./steps/make_mfcc_pitch.sh --nj 30  $folder
        done
fi
