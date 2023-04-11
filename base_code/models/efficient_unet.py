import torch
from segmentation_models_pytorch import Unet


class EfficientUNet2d(torch.nn.Module):
    def __init__(self, cfg):
        super(EfficientUNet2d, self).__init__()
        self.model = Unet(encoder_name = 'efficientnet-b2',
                            encoder_weights = None,
                            in_channels = cfg.DATA.IN_CHANNELS,
                            classes = cfg.DATA.OUT_CHANNELS,
                            activation = None
                            )

    def forward(self, x):
        enc_features = self.model.encoder(x)
        decoder_output = self.model.decoder(*enc_features)
        masks = self.model.segmentation_head(decoder_output)
        return masks
