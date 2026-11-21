import torch
from torchvision.models import resnet34, convnext_base, convnext_large, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
import torch.nn as nn
import os
from models.dino import DINOv2Classifier, DINOv3Classifier
# 修改 models/utils.py 中的 build_model 函数
def build_model(args, device="cuda", num_classes=2):
    if args.model == 'resnet34':
        net = resnet34(pretrained=True)
        in_features = net.fc.in_features
        # 加入 Dropout 防止过拟合
        net.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    elif args.model == 'efficientnet_b3':
        # 尝试使用 torchvision 内置的更强网络
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        net = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif args.model == 'efficientnet_v2_m':
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        net = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif args.model == 'convnext_base':
        net = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        in_features = net.classifier[2].in_features
        net.classifier[2] = nn.Linear(in_features, num_classes)

    elif args.model == 'convnext_large':
        net = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        in_features = net.classifier[2].in_features
        net.classifier[2] = nn.Linear(in_features, num_classes)
    elif args.model == "dinov2":
        net = DINOv2Classifier()
    elif args.model == "dinov3":
        # net = DINOv3Classifier(
        #     model_name="vit_huge_plus_patch16_dinov3.lvd1689m",
        #     ckpt_path = "/data/home/huanghp/hf_models/vit_huge_plus_patch16_dinov3.lvd1689m/model.safetensors")
        net = DINOv3Classifier()
    else:
        raise NotImplementedError
    
    return net.to(device)
def load_model(net, model_path, device="cuda"):
    """
    Load a trained model from a checkpoint file

    Args:
        net: Your model instance (should match the architecture of saved model)
        model_path: Path to the saved model file (.pth or .pt)

    Returns:
        net: Model with loaded weights
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # If the saved file is just state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume it's directly the state_dict
        net.load_state_dict(checkpoint)

    return net.to(device)
def save_model(net, save_path):
    """
    Save model weights to a file

    Args:
        net: Your model instance
        save_path: Path to save the model (.pth or .pt)
    """
    # Save only state_dict (recommended)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net.state_dict(), save_path)