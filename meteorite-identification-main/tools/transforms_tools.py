from PIL import Image, ImageFilter
import random

from torchvision import transforms


class RandomJPEGCompression:
    """Random JPEG compression simulates compression artifacts."""

    def __init__(self, quality_min=25, quality_max=75, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        import io
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownUpSample:
    """Downsample and upsample to simulate low-resolution blur."""

    def __init__(self, scale_min=0.35, scale_max=0.8, p=0.5):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(self.scale_min, self.scale_max)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_small = img.resize((new_w, new_h), Image.BILINEAR)
        return img_small.resize((w, h), Image.BILINEAR)


class RandomGaussianBlur:
    """Random Gaussian blur simulates defocus."""

    def __init__(self, radius_min=0.3, radius_max=1.2, p=0.3):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def build_train_transforms(img_size: int = 224):
    """Combine image and degradation augmentations for Stage 2 training."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            RandomJPEGCompression(quality_min=25, quality_max=75, p=0.5),
            RandomDownUpSample(scale_min=0.35, scale_max=0.8, p=0.5),
            RandomGaussianBlur(radius_min=0.3, radius_max=1.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms(img_size: int = 224):
    """Use deterministic preprocessing for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


