#!/bin/bash
# create environment
conda create -y -n nus08_env python=3.7.6

# install project requirements, conda install if it is available on conda channels, else pip install
while read requirement; do conda install -n nus08_env --y -q -c conda-forge -c pytorch -c anaconda -c ralexx $requirement || pip install $requirement; done < requirements.txt

# activate conda environment
source ~/opt/anaconda3/bin/activate nus08_env

# print python version to ensure it is 3.7.6
python --version 
