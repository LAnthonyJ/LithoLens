import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    一个轻量级 CNN，适合快速冒烟测试与初学者理解。

    设计思路很朴素：
    - 先用卷积层提取局部纹理和边缘特征
    - 再用池化层逐步缩小空间尺寸，减少计算量
    - 最后把整张图压缩成一个较短的特征向量
    - 用线性层输出两个类别的分数

    它不是为了追求最高精度，而是为了：
    - 快速验证数据读取、训练循环、推理流程是否都正常
    - 让初学者更容易理解 CNN 的基础结构
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # features 部分负责“提取图像特征”
        self.features = nn.Sequential(
            # 第 1 个卷积块：
            # 输入是 3 通道 RGB 图像，输出 32 个特征通道
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 第 2 个卷积块：进一步提取更复杂的纹理和形状
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 第 3 个卷积块：把特征通道提升到 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 自适应池化的好处是：
            # 无论输入图片尺寸是多少，最后都统一成 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # classifier 部分负责“根据特征做分类”
        self.classifier = nn.Sequential(
            # 把 [B, 128, 1, 1] 展平成 [B, 128]
            nn.Flatten(),
            # Dropout 可以在一定程度上缓解过拟合
            nn.Dropout(p=0.2),
            # 输出 num_classes 维的 logits
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 前向传播顺序：图像 -> 特征 -> 分类结果
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_simple_cnn(num_classes: int = 2):
    """提供统一的模型构建接口，便于外部按名称调用。"""
    return SimpleCNN(num_classes=num_classes)
