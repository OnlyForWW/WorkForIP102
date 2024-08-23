import timm
from .net import RepMNet
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'resnet50':
        model = timm.create_model('resnet50',
                                  pretrained=False,
                                  drop_path_rate=config.MODEL.DROP_RATE,
                                  num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == 'RepMNet':
        model = RepMNet(drop_path_rate=config.MODEL.DROP_RATE)
    elif model_type == 'RepMNet_deploy':
        model = RepMNet(drop_path_rate=config.MODEL.DROP_RATE, deploy=True)
    else:
        raise NotImplementedError(f'Unknow model: {model_type}')

    return model