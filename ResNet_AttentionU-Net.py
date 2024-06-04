"""These additions were made to the Attention U-Net repository networks.py file to include a ResNet Encoder"""
"""Repo Link: https://github.com/LeeJunHyun/Image_Segmentation"""

class ResNetEncoder(nn.Module):
    def __init__(self, weights=ResNet101_Weights.IMAGENET1K_V1):
        super(ResNetEncoder, self).__init__()
        resnet = resnet101(weights=weights)
        self.initial = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        self.encoder = ResNetEncoder(weights=ResNet101_Weights.IMAGENET1K_V1)

        """Rest of the code is the same"""
