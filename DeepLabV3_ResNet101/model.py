from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101
from torch import nn

def create_deeplab(output_channels=2):
    """Create and modify a DeepLabV3 model to fit our number of classes."""
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)

    # Change the classifier for the number of desired output channels
    model.classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1))
    deeplab = SimpleSegmentationModel(model)
    return deeplab


class SimpleSegmentationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']