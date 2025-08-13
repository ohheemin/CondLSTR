import torch
import torch.nn as nn
from .resnet_backbone import resnet10

class CondLSTR2DRes10(nn.Module):
    def __init__(self, num_classes=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # ResNet10 backbone
        self.backbone = resnet10(norm_layer=norm_layer)
        
        # Feature map size 조정 후 lane head
        self.head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        out = self.head(x)
        return out
