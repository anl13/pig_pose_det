#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python pose_estimation/demo_BamaPig3D.py \
    --dataset_folder /media/AnimalData2/MAMMAL/BamaPig3D/ \
    --cfg experiments/pig/w48_384x384_univ_20210225.yaml \
    --GPUS "(0,)" \
    --vis True \
    --write_json False

