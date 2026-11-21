import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A lightweight CNN for smoke tests and learning the basics.

    Architecture:
    - Use convolutions for local texture and edge features
    - Pool spatial dimensions to reduce computation
    - Compress the image to a short feature vector
    - Use a linear layer for the two class scores

    Use this small network to:
    - check data loading, training and inference
    - illustrate the structure of a CNN
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Extract image features.
        self.features = nn.Sequential(
            # First convolutional block:
            # Map RGB inputs to 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Second block: extract higher-level textures and shapes
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Third block: increase the channel count to 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Adaptive pooling handles different input sizes:
            # Each input is pooled to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Map features to class scores.
        self.classifier = nn.Sequential(
            # Flatten [B, 128, 1, 1] to [B, 128]
            nn.Flatten(),
            # Dropout can reduce overfitting
            nn.Dropout(p=0.2),
            # Return num_classes logits
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Forward pass: image -> features -> predictions
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_simple_cnn(num_classes: int = 2):
    """Construct models by name through a shared interface."""
    return SimpleCNN(num_classes=num_classes)
