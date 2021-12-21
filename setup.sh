#!/bin/bash
if ! [ -x "$(command -v conda)" ]; then
    echo "Error: Please intall conda"
    exit 1 
fi
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n ens python=2.7 # or python=3.6
conda activate ens
# install jupyter notebook
conda install jupyter
