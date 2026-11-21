"""
Meteor feature extraction and SVM/kNN training
Use Meteor-Mamba embeddings in the DINOv2 classifier workflow
"""
import os, sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

DATA_ROOT = "/data/数据科学实践project"
METEOR_PATH = "/tmp/Meteor"
MODEL_DIR = "/data/models"
OUTPUT_DIR = "/data/数据科学实践project/meteor_output"
IMAGE_SIZE = 490

sys.path.insert(0, METEOR_PATH)
device = torch.cuda.current_device()

print("Loading models...")
from meteor.load_mmamba import load_mmamba
from meteor.load_meteor import load_meteor
from utils.utils import freeze_model

mmamba = load_mmamba(f"{MODEL_DIR}/meteor_mamba")
mmamba = mmamba.cuda()
mmamba.mean = mmamba.mean.cuda()
mmamba.std = mmamba.std.cuda()
freeze_model(mmamba)

meteor, tok_meteor = load_meteor(f"{MODEL_DIR}/meteor_mlm", bits=0)
meteor = meteor.cuda()
freeze_model(meteor)

img_token_number = int((490/14)**2)
print("Models loaded!")

# Load train CSV
train_df = pd.read_csv(f"{DATA_ROOT}/train_labels.csv")
train_dir = f"{DATA_ROOT}/train_images/train_images"
print(f"Training samples: {len(train_df)}")


@torch.no_grad()
def extract_features(img_path):
    """Extract Meteor ToR features for a single image"""
    try:
        image = F.interpolate(
            pil_to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic"
        ).squeeze(0)
    except Exception:
        return None

    inputs = [{"image": image, "question": "Is this a meteorite?"}]

    mmamba_inputs = mmamba.eval_process(
        inputs=inputs, tokenizer=tok_meteor, device=device,
        img_token_number=img_token_number
    )
    clip_features = meteor.clip_features(mmamba_inputs["image"])
    mmamba_inputs.update({"image_features": clip_features})
    mmamba_outputs = mmamba(**mmamba_inputs)

    # Return the ToR features averaged over sequence
    tor_features = mmamba_outputs.tor_features
    return tor_features.mean(dim=1).cpu().numpy().flatten()


# Extract training features
print("Extracting training features...")
train_features = []
train_labels = []
failed = 0

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    img_path = f"{train_dir}/{row['id']}"
    feat = extract_features(img_path)
    if feat is not None:
        train_features.append(feat)
        train_labels.append(int(row["label"]))
    else:
        failed += 1

train_X = np.array(train_features)
train_y = np.array(train_labels)
print(f"Features: {train_X.shape}, Labels: {np.bincount(train_y)}, Failed: {failed}")

# 5-fold CV
print("\nTraining classifiers with 5-fold CV...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0,
                        class_weight="balanced", probability=True, random_state=42))
    svm.fit(X_tr, y_tr)
    print(f"Fold {fold+1}: SVM val acc = {svm.score(X_val, y_val):.4f}")

    knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
    knn.fit(X_tr, y_tr)
    print(f"Fold {fold+1}: kNN val acc = {knn.score(X_val, y_val):.4f}")

# Train final classifiers
print("\nTraining final classifiers on all data...")
final_svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0,
                          class_weight="balanced", probability=True, random_state=42))
final_svm.fit(train_X, train_y)

final_knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
final_knn.fit(train_X, train_y)

with open(f"{OUTPUT_DIR}/meteor_classifiers.pkl", "wb") as f:
    pickle.dump({"svm": final_svm, "knn": final_knn}, f)

# Extract test features
print("\nExtracting test features...")
test_df = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
test_dir = f"{DATA_ROOT}/test_images/test_images"

test_features = []
test_ids = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_path = f"{test_dir}/{row['id']}"
    feat = extract_features(img_path)
    if feat is not None:
        test_features.append(feat)
        test_ids.append(row["id"])
    else:
        test_features.append(np.zeros(train_X.shape[1]))
        test_ids.append(row["id"])

test_X = np.array(test_features)
test_X = normalize(test_X, norm="l2", axis=1)

# Predict
svm_probs = final_svm.predict_proba(test_X)[:, 1]
knn_probs = final_knn.predict_proba(test_X)[:, 1]
final_scores = (svm_probs + knn_probs) / 2.0

# Save
results_df = pd.DataFrame({"id": test_ids, "prob": final_scores})
results_df = results_df.sort_values("prob", ascending=False)
results_df.to_csv(f"{OUTPUT_DIR}/meteor_features_probs.csv", index=False)

# Top-85 submission
top_k = 85
results_df["label"] = 0
results_df.iloc[:top_k, results_df.columns.get_loc("label")] = 1

template = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
template["label"] = template["id"].map(results_df.set_index("id")["label"])
template["label"] = template["label"].fillna(0).astype(int)
template.to_csv(f"{OUTPUT_DIR}/submission_meteor_features_top85.csv", index=False)

print(f"\nDone! Top-85 submission saved.")
print(f"SVM+KNN ensemble predictions: top {top_k} as meteorites")
