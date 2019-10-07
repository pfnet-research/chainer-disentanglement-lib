#!/usr/bin/env bash

#Set up training environment.
export EVALUATION_NAME=dev_trial
export DATASET_NAME=dsprites_full

export NDC_ROOT=.
export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}

# You can change output path & dataset path
# DISENTANGLEMENT_LIB_DATA will be used to qualitative evaluation
export OUTPUT_PATH=./results
export DISENTANGLEMENT_LIB_DATA=./dataset