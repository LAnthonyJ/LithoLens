"""
Multi-model zero-shot meteorite classification on GPU server
Models: CLIP ViT-L/14, SigLIP, DINOv2 + classifier
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Paths
TEST_DIR = Path('/data/数据科学实践project/test_images/test_images')
TEMPLATE_CSV = Path('/data/数据科学实践project/sample_submission.csv')
OUTPUT_DIR = Path('/data/数据科学实践project/meteor_zs_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==========================================
# Prompt pairs for ensemble
# ==========================================
PROMPT_PAIRS = [
    ("a photo of a meteorite, a rock from outer space with dark fusion crust",
     "a photo of an ordinary terrestrial rock or stone found on earth"),
    ("a meteorite specimen showing fusion crust and regmaglypts",
     "a common earth rock like granite, basalt, or sandstone"),
    ("an iron or stony meteorite from space, possibly with thumbprint-like textures",
     "a regular rock, pebble, or mineral found on the ground"),
    ("a chondrite meteorite with visible chondrules and dark matrix",
     "a piece of slag, industrial byproduct, or ordinary stone"),
]

# ==========================================
# 1. CLIP ViT-L/14 Zero-Shot
# ==========================================
print("\n" + "="*60)
print("MODEL 1: CLIP ViT-L/14 Zero-Shot")
print("="*60)

from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

@torch.no_grad()
def clip_predict(image_paths, batch_size=16):
    all_probs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP ViT-L/14"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                batch_imgs.append(Image.new("RGB", (224, 224)))

        batch_probs = []
        for meteor_prompt, rock_prompt in PROMPT_PAIRS:
            inputs = clip_processor(
                text=[meteor_prompt, rock_prompt],
                images=batch_imgs, return_tensors="pt", padding=True
            ).to(DEVICE)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
            batch_probs.append(probs)
        all_probs.append(np.mean(batch_probs, axis=0))
    return np.concatenate(all_probs)

image_files = sorted(TEST_DIR.glob("*.jpg"))
image_paths = [str(p) for p in image_files]
image_ids = [p.name for p in image_files]

clip_probs = clip_predict(image_paths)
print(f"CLIP prob range: [{clip_probs.min():.4f}, {clip_probs.max():.4f}]")
print(f"CLIP >0.5: {(clip_probs > 0.5).sum()}")

# ==========================================
# 2. SigLIP Zero-Shot
# ==========================================
print("\n" + "="*60)
print("MODEL 2: SigLIP SO400M Zero-Shot")
print("="*60)

from transformers import AutoProcessor, AutoModel

siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(DEVICE)
siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
siglip_model.eval()

@torch.no_grad()
def siglip_predict(image_paths, batch_size=16):
    all_probs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="SigLIP"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                batch_imgs.append(Image.new("RGB", (384, 384)))

        batch_probs = []
        for meteor_prompt, rock_prompt in PROMPT_PAIRS:
            inputs = siglip_processor(
                text=[meteor_prompt, rock_prompt],
                images=batch_imgs, return_tensors="pt", padding="max_length"
            ).to(DEVICE)
            outputs = siglip_model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.sigmoid(logits[:, 0] - logits[:, 1]).cpu().numpy()
            batch_probs.append(probs)
        all_probs.append(np.mean(batch_probs, axis=0))
    return np.concatenate(all_probs)

siglip_probs = siglip_predict(image_paths)
print(f"SigLIP prob range: [{siglip_probs.min():.4f}, {siglip_probs.max():.4f}]")
print(f"SigLIP >0.5: {(siglip_probs > 0.5).sum()}")

# ==========================================
# 3. DINOv2 ViT-L14 + kNN (lightweight classifier)
# ==========================================
print("\n" + "="*60)
print("MODEL 3: DINOv2 ViT-L14 Features + Centroid")
print("="*60)

# Load training data
train_csv = pd.read_csv('/data/数据科学实践project/train_labels.csv')
train_dir = Path('/data/数据科学实践project/train_images/train_images')

# Load DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(DEVICE)
dinov2.eval()

@torch.no_grad()
def dinov2_extract(paths, batch_size=8):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    features = []
    for i in tqdm(range(0, len(paths), batch_size), desc="DINOv2"):
        batch_imgs = []
        valid_idx = []
        for j, p in enumerate(paths[i:i+batch_size]):
            try:
                img = Image.open(p).convert("RGB")
                img = T.ToTensor()(img)
                img = transform(img)
                batch_imgs.append(img)
                valid_idx.append(j)
            except Exception:
                continue
        if not batch_imgs:
            continue
        imgs = torch.stack(batch_imgs).to(DEVICE)
        feats = dinov2(imgs).cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        features.append(feats)
    return np.concatenate(features)

# Extract DINOv2 features for training data (sample 1000 for speed)
print("Extracting DINOv2 training features...")
train_sample = train_csv.sample(min(1000, len(train_csv)), random_state=42)
train_paths_sample = [str(train_dir / rid) for rid in train_sample['id']]
train_labels_sample = train_sample['label'].values

train_feats = dinov2_extract(train_paths_sample)

# Compute centroids
pos_centroid = train_feats[train_labels_sample == 1].mean(axis=0)
neg_centroid = train_feats[train_labels_sample == 0].mean(axis=0)

print("Extracting DINOv2 test features...")
test_feats = dinov2_extract(image_paths)

# Cosine similarity to centroids
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

dino_probs = np.array([
    cosine_sim(f, pos_centroid) / (cosine_sim(f, pos_centroid) + cosine_sim(f, neg_centroid) + 1e-8)
    for f in test_feats
])
print(f"DINOv2 prob range: [{dino_probs.min():.4f}, {dino_probs.max():.4f}]")
print(f"DINOv2 >0.5: {(dino_probs > 0.5).sum()}")

# ==========================================
# 4. Ensemble
# ==========================================
print("\n" + "="*60)
print("ENSEMBLE RESULTS")
print("="*60)

# Normalize each model's scores to [0,1] using min-max
def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

clip_norm = minmax_norm(clip_probs)
siglip_norm = minmax_norm(siglip_probs)
dino_norm = minmax_norm(dino_probs)

# Weighted ensemble (CLIP and SigLIP weighted more as they're zero-shot)
ensemble_probs = 0.35 * clip_norm + 0.35 * siglip_norm + 0.30 * dino_norm

results = []
for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
    n = int((ensemble_probs > threshold).sum())
    results.append((threshold, n))
    print(f"  Ensemble threshold {threshold}: {n} meteorites")

# Use 0.5 threshold for final
final_labels = (ensemble_probs > 0.5).astype(int)
print(f"\nFinal (threshold=0.5): {final_labels.sum()} meteorites")

# ==========================================
# Save results
# ==========================================
print("\n=== Saving ===")

# Individual model results
for name, probs in [("clip_vitl14", clip_probs), ("siglip", siglip_probs),
                     ("dinov2", dino_probs), ("ensemble", ensemble_probs)]:
    labels = (probs > 0.5).astype(int)
    df = pd.DataFrame({'id': image_ids, 'prob': probs, 'label': labels})
    df.to_csv(OUTPUT_DIR / f'pred_{name}.csv', index=False)
    print(f"  {name}: {labels.sum()} meteorites -> pred_{name}.csv")

# Submission file (ensemble)
template = pd.read_csv(TEMPLATE_CSV)
sub = template[['id']].copy()
sub['label'] = sub['id'].map(
    dict(zip(image_ids, final_labels))
).fillna(0).astype(int)
sub.to_csv(OUTPUT_DIR / 'submission_ensemble_3model.csv', index=False)
print(f"\nSubmission: {OUTPUT_DIR / 'submission_ensemble_3model.csv'}")
print(f"Meteorites: {sub['label'].sum()}/{len(sub)}")

# Save all probabilities
probs_df = pd.DataFrame({
    'id': image_ids,
    'clip_vitl14': clip_probs,
    'siglip': siglip_probs,
    'dinov2': dino_probs,
    'ensemble': ensemble_probs,
})
probs_df.to_csv(OUTPUT_DIR / 'all_model_probs.csv', index=False)

elapsed = time.time() - globals().get('_start_time', time.time())
print(f"\nDone!")
