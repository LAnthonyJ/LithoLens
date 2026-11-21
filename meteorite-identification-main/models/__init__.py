from .resnet18_model import build_resnet18
from .simple_cnn import build_simple_cnn


def build_model(name: str, num_classes: int = 2, pretrained: bool = False):
    """统一的模型工厂函数，便于在命令行中切换模型。"""
    name = name.lower().strip()
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if name == "simple_cnn":
        return build_simple_cnn(num_classes=num_classes)
    raise ValueError(f"不支持的模型名称: {name}")
