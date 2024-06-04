from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[2:]
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        x5 = F.relu(self.bn5(self.conv5(self.avg_pool(x))))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x_out = F.relu(self.bn_out(self.conv_out(x_cat)))
        return x_out

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, 256)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, features):
        low_level_feature = self.project(features['low_level'])  # Extract low-level features correctly
        x = self.aspp(features['out'])
        x = F.interpolate(x, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feature], dim=1)
        x = self.classifier(x)
        return x

class DeepLabV3PlusModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def create_deeplabv3plus(output_channels=2):
    """Create and modify a DeepLabV3+ model to fit our number of classes."""
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    backbone = deeplabv3_resnet101(weights=weights).backbone

    # Use IntermediateLayerGetter to get the required intermediate features
    return_layers = {'layer1': 'low_level', 'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3PlusModel(
        backbone,
        DeepLabHeadV3Plus(2048, 256, output_channels)
    )

    return model

