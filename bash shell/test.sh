#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py \
       --cfg config.yaml --data-path ./dataset \
       --eval \
       --deploy \
       --pretrained ./output/repNet/exp2/ckpt_epoch_279.pth
