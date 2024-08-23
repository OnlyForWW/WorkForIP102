#!/bin/bash
python erf/visualize_erf_2.py --model repmnet --weights ./erf/rep_1k_pretrained.pth --data_path ./ImageNet-1K/ --save_path repmnet_erf_matrix.npy
python erf/analyze_erf.py --source repmnet_erf_matrix.npy --heatmap_save repmnet_heatmap.png