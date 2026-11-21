"""
Pure CLIP zero-shot classification - no training, no classifier
Uses multiple prompt pairs and calibrated threshold from training data
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"H:\数据科学与大数据技术\数据科学实践\project1\project_stage2")

# Find image dirs
def find_dir(root, name):
    d = root / f"{name}_images"
    if d.is_dir():
        inner = list(d.iterdir())
        if len(inner) == 1 and inner[0].is_dir():
            return inner[0]
        return d
    return d

TRAIN_DIR = find_dir(ROOT, "train")
TEST_DIR = find_dir(ROOT, "test")
print(f"Train: {TRAIN_DIR}")
print(f"Test: {TEST_DIR}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load CLIP
from transformers import CLIPProcessor, CLIPModel
model_id = "openai/clip-vit-base-patch32"
print(f"Loading {model_id}...")
model = CLIPModel.from_pretrained(model_id).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

# Diverse prompt pairs for ensemble
PROMPT_PAIRS = [
    ("a photo of a meteorite, a rock from outer space showing dark fusion crust",
     "a photo of an ordinary terrestrial rock or stone found on earth"),
    ("a meteorite specimen with regmaglypts and dark crust, possibly iron or chondrite",
     "a common rock, pebble, mineral, or stone from the ground"),
    ("a space rock from asteroid or meteor, showing signs of atmospheric entry",
     "sandstone, granite, basalt, limestone, or other earth rock"),
    ("a meteorite fragment collected by scientists, dark and rounded with fusion crust",
     "an ordinary stone, gravel, concrete, slag, or industrial material"),
    ("a genuine meteorite with possible chondrules, metal flakes, or fusion crust",
     "a terrestrial rock with weathering, crystals, or sedimentary layers"),
]

@torch.no_grad()
def zero_shot_predict(image_paths, model, processor, device, batch_size=16):
    """Zero-shot meteorite probability using prompt ensemble"""
    all_probs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Zero-Shot"):
        batch = image_paths[i:i+batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except:
                imgs.append(Image.new("RGB", (224, 224)))

        batch_probs = []
        for met_prompt, rock_prompt in PROMPT_PAIRS:
            inputs = processor(
                text=[met_prompt, rock_prompt],
                images=imgs,
                return_tensors="pt",
                padding=True
            ).to(device)
            outputs = model(**inputs)
            # logits_per_image: [batch, 2], column 0 = meteorite, column 1 = rock
            probs = outputs.logits_per_image.softmax(dim=-1)[:, 0].cpu().numpy()
            batch_probs.append(probs)
        # Average across all prompt pairs
        all_probs.append(np.mean(batch_probs, axis=0))
    return np.concatenate(all_probs)

# Run on training data to find optimal threshold
print("\n=== Calibrating threshold on training data ===")
df_train = pd.read_csv(ROOT / "train_labels.csv")

# Use a subset for calibration (first 1000 images)
df_calib = df_train.sample(min(1000, len(df_train)), random_state=42)
calib_paths = [str(TRAIN_DIR / rid) for rid in df_calib['id']]
calib_labels = df_calib['label'].values.astype(int)

# Filter existing
exist = [Path(p).exists() for p in calib_paths]
calib_paths = [p for p, e in zip(calib_paths, exist) if e]
calib_labels = calib_labels[exist]
print(f"Calibration set: {len(calib_paths)} images")

calib_probs = zero_shot_predict(calib_paths, model, processor, DEVICE)

# Find optimal threshold maximizing F1
best_threshold, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.025):
    preds = (calib_probs > t).astype(int)
    tp = ((preds == 1) & (calib_labels == 1)).sum()
    fp = ((preds == 1) & (calib_labels == 0)).sum()
    fn = ((preds == 0) & (calib_labels == 1)).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.3f} (F1={best_f1:.3f})")

# Evaluate on calibration set
preds = (calib_probs > best_threshold).astype(int)
acc = (preds == calib_labels).mean()
tp = ((preds == 1) & (calib_labels == 1)).sum()
fp = ((preds == 1) & (calib_labels == 0)).sum()
fn = ((preds == 0) & (calib_labels == 1)).sum()
tn = ((preds == 0) & (calib_labels == 0)).sum()
print(f"Calibration results: Acc={acc:.3f}, F1={best_f1:.3f}")
print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"  Predicted meteorites: {preds.sum()} / {len(preds)}")

# Run on test data
print(f"\n=== Running on test data ===")
test_ids = pd.read_csv(ROOT / "sample_submission.csv")['id'].tolist()
test_paths = [str(TEST_DIR / rid) for rid in test_ids]
exist_test = [Path(p).exists() for p in test_paths]
test_paths = [p for p, e in zip(test_paths, exist_test) if e]
test_ids_valid = [rid for rid, e in zip(test_ids, exist_test) if e]
print(f"Test images: {len(test_paths)}")

test_probs = zero_shot_predict(test_paths, model, processor, DEVICE)

# Apply calibrated threshold
test_labels = (test_probs > best_threshold).astype(int)
n_meteor = int(test_labels.sum())

print(f"\n=== RESULTS ===")
print(f"Threshold: {best_threshold:.3f}")
print(f"Meteorites: {n_meteor}")
print(f"Non-meteorites: {len(test_labels) - n_meteor}")
print(f"Prob range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

# Top predictions
order = np.argsort(-test_probs)
print(f"\nTop-15 meteorites:")
for i in order[:15]:
    print(f"  {test_ids_valid[i]}: {test_probs[i]:.4f}")
print(f"\nBottom-15 (likely non-meteorites):")
for i in order[-15:]:
    print(f"  {test_ids_valid[i]}: {test_probs[i]:.4f}")

# Save results
template = pd.read_csv(ROOT / "sample_submission.csv")
sub = template[['id']].copy()
sub['label'] = sub['id'].map(dict(zip(test_ids_valid, test_labels))).fillna(0).astype(int)
sub.to_csv(ROOT / 'submission_clip_zeroshot_calibrated.csv', index=False)

prob_df = pd.DataFrame({'id': test_ids_valid, 'prob': test_probs})
prob_df.to_csv(ROOT / 'test_probs_clip_zeroshot.csv', index=False)

print(f"\nSaved: submission_clip_zeroshot_calibrated.csv")
print(f"Saved: test_probs_clip_zeroshot.csv")
