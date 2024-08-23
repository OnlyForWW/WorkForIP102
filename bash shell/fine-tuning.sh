#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py \
       --cfg ft.yaml --data-path ./dataset \
       --finetune ./checkpoint-best.pth
