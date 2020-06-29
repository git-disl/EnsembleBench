#!/bin/bash
if ! [ -x "$(command -v conda)" ]; then
    echo "Error: Please intall conda"
    exit 1 
fi
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n pytorch python=2.7 # or python=3.6
conda activate pytorch
conda install pytorch=0.4.1 cuda80 -c pytorch
conda install torchvision
# install jupyter notebook
conda install jupyter
