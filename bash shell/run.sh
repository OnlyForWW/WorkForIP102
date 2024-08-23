#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py \
       --cfg config.yaml --data-path ./dataset \
#       --pretrained ./resnet50_1kpretrained_timm_style.pth
