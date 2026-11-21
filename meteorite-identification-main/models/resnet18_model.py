import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = False):
    """
    Build a ResNet18 binary classifier.

    settings:
    - num_classes: Output classes. Binary classification defaults to 2.
    - pretrained: Load ImageNet pretrained weights.
      When enabled, training starts from general visual features,
      which can stabilize early training.
    """
    # Load the default pretrained weights when pretrained=True;
    # Otherwise initialize the model randomly.
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # ResNet18 was designed for 1,000 classes,
    # so the original fully connected layer has 1,000 outputs.
    # Replace that layer for binary classification.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
