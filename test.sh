#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python pose_estimation/test.py \
    --cfg experiments/pig/w48_384x384_univ_20210225.yaml
