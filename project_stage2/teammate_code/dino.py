import torch
import torch.nn as nn
import timm
class DINOv2Classifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name="vit_base_patch14_dinov2.lvd142m",  # ViT-B/14
        freeze=True,
        head="mlp",          # "linear" or "mlp"
        img_size=448,        # must be a multiple of 14
    ):
        super().__init__()
        self.freeze = freeze

        # num_classes=0 -> remove the head and return pooled features of shape (B, feat_dim)
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=448,
            pretrained_cfg_overlay={
                "file": "/data/home/huanghp/hf_models/vit_base_patch14_dinov2.lvd142m/model.safetensors"
            },
        )

        feat_dim = self.backbone.num_features  # ViT-B = 768

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if head == "linear":
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():        # skip the gradient graph to save memory and compute
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self.head(feat)

    # net.train() runs during training; keep the frozen backbone in eval
    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self
import os
class DINOv3Classifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        freeze=True,
        head="mlp",          # "linear" or "mlp"
        img_size=512,        # DINOv3 ViT-L/16 defaults to 256; must be a multiple of 16
        model_name="vit_large_patch16_dinov3.lvd1689m",
        ckpt_path="/data/home/huanghp/hf_models/vit_large_patch16_dinov3.lvd1689m/model.safetensors",
    ):
        super().__init__()

        assert img_size % 16 == 0, "DINOv3 ViT-L/16 requires img_size to be divisible by 16."
        self.freeze = freeze

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Please download it first with:\n"
                f"hf download timm/{model_name} --local-dir {os.path.dirname(ckpt_path)}"
            )

        # num_classes=0: remove classification head, output image embedding
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
            pretrained_cfg_overlay={
                "file": ckpt_path
            },
        )

        feat_dim = self.backbone.num_features  # DINOv3 ViT-L = 1024

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if head == "linear":
            self.head = nn.Linear(feat_dim, num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        elif head == "mlp_large":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 2048),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 512),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head}. Use 'linear' or 'mlp'.")

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)

        logits = self.head(feat)
        return logits

    def train(self, mode=True):
        super().train(mode)

        # When freeze=True, net.train() keeps the backbone in eval
        if self.freeze:
            self.backbone.eval()

        return self