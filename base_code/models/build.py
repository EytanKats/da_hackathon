import torch.nn as nn
import torchvision
from base_code.models.unet import UNet2d
from base_code.models.efficient_unet import EfficientUNet2d


def build_model(cfg):
    arch = cfg.MODEL.ARCHITECTURE
    if arch == 'UNet2d':
        model = UNet2d(cfg)
    elif arch == 'EfficientUNet2d':
        model = EfficientUNet2d(cfg)
    else:
        raise ValueError()

    return model