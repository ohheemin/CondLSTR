import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

def resnet10(pretrained=False, norm_layer=nn.BatchNorm2d):
    """
    ResNet10 backbone
    layers=[1,1,1,1] -> ResNet10
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1000, norm_layer=norm_layer)
    if pretrained:
        raise NotImplementedError("ResNet10 pretrained not available")
    return model
