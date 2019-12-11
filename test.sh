#!/bin/bash 

CUDA_VISIBLE_DEVICES=4 python pose_estimation/test.py \
    --cfg experiments/pig/resnet152/288x384_d256x3_adam_lr1e-3.yaml
