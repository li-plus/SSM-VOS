import torch
from torch import nn
from torchvision import models


class ResNet(models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        del self.avgpool
        del self.fc

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5, x4, x3, x2, x1


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        state_dict = models.resnet._resnet(arch, block, layers, pretrained, progress, **kwargs).state_dict()
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        state_dict['conv1.weight'] = torch.cat(
            [state_dict['conv1.weight'], torch.FloatTensor(64, 1, 7, 7).normal_(0, 0.0001)], dim=1)
        model.load_state_dict(state_dict)

    return model


def resnet50(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet50', models.resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet101', models.resnet.Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
