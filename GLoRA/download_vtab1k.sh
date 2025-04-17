#!/bin/bash

# Create directory for VTAB-1K dataset
mkdir -p /l/users/mukul.ranjan/glora/data
cd /l/users/mukul.ranjan/glora/data

echo "Downloading VTAB-1K dataset (4.66GB) from Hugging Face..."
wget -O vtab-1k.zip https://huggingface.co/datasets/XiN0919/VTAB-1k/resolve/main/vtab-1k.zip?download=true

echo "Unzipping VTAB-1K dataset..."
unzip vtab-1k.zip

echo "Cleaning up..."
# Remove the zip file to save space after extraction
rm vtab-1k.zip

echo "VTAB-1K dataset downloaded and extracted to /l/users/mukul.ranjan/glora/data/vtab-1k"
echo "You can now run the evaluation script: ./eval_vtab.sh"
