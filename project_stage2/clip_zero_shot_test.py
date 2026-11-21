"""
CLIP zero-shot test on training data to verify meteorite detection ability
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

# Paths
ROOT = Path(r"H:\数据科学与大数据技术\数据科学实践\project1\project_stage2")
TRAIN_CSV = ROOT / "train_labels.csv"

# Check actual train_images structure
train_root = ROOT / "train_images"
if train_root.is_dir():
    inner = list(train_root.iterdir())
    if len(inner) == 1 and inner[0].is_dir():
        TRAIN_DIR = inner[0]
    else:
        TRAIN_DIR = train_root
else:
    TRAIN_DIR = train_root
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"TRAIN_DIR exists: {TRAIN_DIR.exists()}")
TEST_DIR = ROOT / "test_images" / "test_images"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load CLIP
from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
print(f"Loading {model_id}...")
model = CLIPModel.from_pretrained(model_id).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()
print("Model loaded!")

# Multi-prompt ensemble for better accuracy
PROMPT_PAIRS = [
    ("a photo of a meteorite, a rock from outer space", "a photo of an ordinary terrestrial rock"),
    ("a meteorite with dark fusion crust", "a common earth rock or stone"),
    ("a space rock, meteorite specimen", "a regular rock found on the ground"),
    ("an iron or stony meteorite", "sandstone, granite, or basalt rock"),
]

@torch.no_grad()
def predict_meteorite(image_path, model, processor, device):
    """Ensemble prediction using multiple prompt pairs"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return 0, 0.0

    meteorite_probs = []

    for meteor_prompt, rock_prompt in PROMPT_PAIRS:
        inputs = processor(
            text=[meteor_prompt, rock_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # [2]
        probs = logits.softmax(dim=-1)
        meteorite_probs.append(probs[0].item())

    avg_prob = np.mean(meteorite_probs)
    label = 1 if avg_prob > 0.5 else 0
    return label, avg_prob


# Test on training samples
print("\n=== Training Set Sample Test ===")
df_train = pd.read_csv(TRAIN_CSV)
print(f"Training set: {len(df_train)} images, label distribution:\n{df_train['label'].value_counts().sort_index()}")

# Sample 10 from each class
train_meteor = df_train[df_train['label'] == 1].sample(10, random_state=42)
train_rock = df_train[df_train['label'] == 0].sample(10, random_state=42)
test_samples = pd.concat([train_meteor, train_rock])

correct = 0
for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Testing"):
    img_path = TRAIN_DIR / row['id']
    if not img_path.exists():
        print(f"  Missing: {img_path}")
        continue
    pred, prob = predict_meteorite(img_path, model, processor, DEVICE)
    true_label = row['label']
    if pred == true_label:
        correct += 1
    status = "OK" if pred == true_label else "WRONG"
    print(f"  {row['id']} true={true_label} pred={pred} prob={prob:.3f} [{status}]")

print(f"\nAccuracy on 20 samples: {correct}/20 = {correct/20:.1%}")
