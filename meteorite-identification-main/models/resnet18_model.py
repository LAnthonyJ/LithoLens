import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = False):
    """
    构建 ResNet18 二分类模型。

    参数:
    - num_classes: 最终输出类别数。当前项目是二分类，所以默认是 2。
    - pretrained: 是否加载 ImageNet 预训练权重。
      如果为 True，模型会从已经学到的通用视觉特征开始训练，
      往往比从零开始更稳定。
    """
    # 如果 pretrained=True，就加载官方默认预训练权重；
    # 否则从随机初始化开始训练。
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # ResNet18 原始版本是做 1000 分类的，
    # 因此最后一层全连接层输出 1000 维。
    # 我们这里只做二分类，所以把最后一层替换掉。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
