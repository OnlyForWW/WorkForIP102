# PestScope: A High-performance Agricultural Pest Identification Framework

## Contributors

[**Junyan Wang**](https://github.com/OnlyForWW), Internet School of Anhui University, Heifei, China.

[**Haoran Yu**](https://github.com/yhrzzz), Queen Mary University of London, London, the UK.

**Feiyang Kang**, Institute of Advanced Technology, University of Science and Technology of China, Hefei, 230031, China.

## Overall Architecture

### Convolutional neural network: RepMSNet

**No pre-training:**

Weight Download: [**No-Rep-weight**](https://pan.baidu.com/s/1cDx_0dVS4QhsTisLkDD2wA?pwd=zgyp)/[**Rep-weight**](https://pan.baidu.com/s/1cDx_0dVS4QhsTisLkDD2wA?pwd=zgyp)

|     Models      | Params (M) | FLOPs (G) | Top1 acc |    FPS     |
| :-------------: | :--------: | :-------: | :------: | :--------: |
|    ResNet-50    |    23.7    |    4.1    |   68.7   |   119.7    |
| EfficientNet-B1 |    6.7     |    0.6    |   66.3   |    62.2    |
|    RepVGG-B0    |    14.7    |    3.4    |   67.2   | 95.6/249.4 |
|   ConvNext-T    |    27.9    |    4.5    |   68.6   |   101.6    |
|   FasterNet-S   |    30.0    |    4.6    |   66.1   |   110.9    |
|     Swin-T      |    27.6    |    4.5    |   69.2   |    80.5    |
|     CSwin-T     |    21.9    |    4.3    |   69.0   |    26.1    |
|     MPViT-S     |    22.6    |    4.8    |   70.4   |    22.2    |
|   Biformer-S    |    25.0    |    4.5    |   71.6   |    26.9    |
|    DWVIT-ES     |    19.6    |    3.5    |   71.6   |     -      |
|  RepMNet (our)  |    28.8    |    4.5    | **72.0** | 77.9/134.4 |

**Pre-training:**

ImageNet-1K Pre-training Weight Download: [**Pre-training-weight**](https://pan.baidu.com/s/1cDx_0dVS4QhsTisLkDD2wA?pwd=zgyp)

Weight Download: [**No-Rep-weight**](https://pan.baidu.com/s/1cDx_0dVS4QhsTisLkDD2wA?pwd=zgyp)/[**Rep-weight**](https://pan.baidu.com/s/1cDx_0dVS4QhsTisLkDD2wA?pwd=zgyp)

|  **Models**   | **Params(M)** | **FLOPs(G)** | **Top1 acc** |  **FPS**   |
| :-----------: | :-----------: | :----------: | :----------: | :--------: |
|   ResNet-50   |     23.7      |     4.1      |     72.8     |   119.7    |
| EfficientNet  |      6.7      |     0.6      |     73.4     |    62.2    |
|   RepVGG-B0   |     14.7      |     3.4      |     72.4     | 95.6/249.4 |
|  ConvNext-T   |     27.9      |     4.5      |     73.8     |   101.6    |
|  FasterNet-S  |     30.0      |     4.6      |     76.0     |   110.9    |
|    Swin-T     |     27.6      |     4.5      |     75.9     |    80.5    |
|    CSwin-T    |     21.9      |     4.3      |     75.0     |    26.1    |
|    MPViT-S    |     22.6      |     4.8      |     75.7     |    22.2    |
|  BiFormer-S   |     25.0      |     4.5      |     76.1     |    26.9    |
|   MIXTURE_1   |     85.8      |     17.6     |     74.7     |     -      |
|   MIXTURE_2   |     28.3      |      -       |     74.9     |     -      |
|   DWVIT-ES    |     19.6      |     3.5      |     76.0     |     -      |
| RepMNet (our) |     28.8      |     4.5      |   **76.3**   | 77.9/134.4 |

## How to Use Our Code

## Acknowledgement

- [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [DingXiaoH/RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch)
- [bethgelab/model-vs-human](https://github.com/bethgelab/model-vs-human)
- [timm](https://github.com/huggingface/pytorch-image-models)
