from fastai.vision.all import unet_learner, resnet50, resnet34
from fastai.metrics import Dice

def create_unet(dls, output_channels=2):
    return unet_learner(dls, resnet34, n_out=output_channels, pretrained=True, metrics=[Dice()])
