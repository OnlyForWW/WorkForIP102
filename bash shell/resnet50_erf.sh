#!/bin/bash
python erf/visualize_erf.py --model resnet50 --weights ./erf/resnet50-0676ba61.pth --data_path ./ImageNet-1K/ --save_path resnet50_erf_matrix.npy
python erf/analyze_erf.py --source resnet50_erf_matrix.npy --heatmap_save resnet50_heatmap.png