#!/usr/bin/env bash

# Siddharth Pittie (sp4013@columbia.edu)

mkdir -p db

BUCKET_NAME="cu-multilang-dataset"

cd db/

echo "Starting download..."

gsutil -m cp -r gs://$BUCKET_NAME ./

echo "Download complete."
