#!/bin/bash

# Get pre-trained model checkpoints for testing
CHECKPOINT_DIR=./checkpoints/battery_cathode/disentangle
mkdir -p "$CHECKPOINT_DIR"
wget -O "$CHECKPOINT_DIR"/disentangle_Disc.pt https://www.dropbox.com/s/ny7fjixxkbibgdv/disentangle_Disc.pt?dl=0
wget -O "$CHECKPOINT_DIR"/disentangle_Gen.pt https://www.dropbox.com/s/7tgsjagu2d0w5vq/disentangle_Gen.pt?dl=0
wget -O "$CHECKPOINT_DIR"/disentangle_params.data https://www.dropbox.com/s/mgv9xcobesww0ia/disentangle_params.data?dl=0

# Download training data
DATA_DIR=./data/battery_cathode
mkdir -p "$DATA_DIR"
wget -O "$DATA_DIR"/synth_data.zip https://www.dropbox.com/s/v4w2xl0fs3hh70g/synth_data.zip?dl=0
unzip -d "$DATA_DIR"/ "$DATA_DIR"/synth_data.zip

# Creating conda environment, downloading requirements
conda create -n sliceGAN python=3.7
conda activate sliceGAN
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=sliceGAN
pip install -r requirements.txt