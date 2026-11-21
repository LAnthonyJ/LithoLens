"""Generate submissions from best classifiers"""
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

DATA_ROOT = Path("/data/数据科学实践project")
FEATURES_DIR = DATA_ROOT / "runs" / "dinov2_features"
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = DATA_ROOT / "runs" / f"classifier_top3_{ts}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

train_data = np.load(FEATURES_DIR / "train_features.npz")
test_data = np.load(FEATURES_DIR / "test_features.npz")
train_X, train_y = train_data["X"], train_data["y"].astype(int)
test_X = test_data["X"]
test_ids = list(test_data["ids"])
clean_ids = [Path(p).name for p in test_ids]

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

predictions = {}

# --- LogisticRegression ---
print("LogisticRegression...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
test_pred = np.zeros(len(test_X))
val_f1s = []
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    clf.fit(X_tr, y_tr)
    val_pred = clf.predict(X_val)
    tp = ((val_pred==1) & (y_val==1)).sum()
    fp = ((val_pred==1) & (y_val==0)).sum()
    fn = ((val_pred==0) & (y_val==1)).sum()
    val_f1s.append(2*tp/max(2*tp+fp+fn, 1))
    test_pred += clf.predict_proba(test_X)[:,1] / n_splits
predictions["LogisticRegression"] = test_pred
print(f"  Val F1: {np.mean(val_f1s):.4f}")

# --- LinearSVC ---
print("LinearSVC...")
clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=3000, random_state=SEED))
test_pred = np.zeros(len(test_X))
val_f1s = []
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    clf.fit(X_tr, y_tr)
    val_pred = clf.predict(X_val)
    tp = ((val_pred==1) & (y_val==1)).sum()
    fp = ((val_pred==1) & (y_val==0)).sum()
    fn = ((val_pred==0) & (y_val==1)).sum()
    val_f1s.append(2*tp/max(2*tp+fp+fn, 1))
    d = clf.decision_function(test_X)
    test_pred += (1/(1+np.exp(-d))) / n_splits
predictions["LinearSVC"] = test_pred
print(f"  Val F1: {np.mean(val_f1s):.4f}")

# --- kNN ---
print("kNN_k25...")
clf = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")
test_pred = np.zeros(len(test_X))
val_f1s = []
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    clf.fit(X_tr, y_tr)
    val_pred = clf.predict(X_val)
    tp = ((val_pred==1) & (y_val==1)).sum()
    fp = ((val_pred==1) & (y_val==0)).sum()
    fn = ((val_pred==0) & (y_val==1)).sum()
    val_f1s.append(2*tp/max(2*tp+fp+fn, 1))
    test_pred += clf.predict_proba(test_X)[:,1] / n_splits
predictions["kNN_k25"] = test_pred
print(f"  Val F1: {np.mean(val_f1s):.4f}")

# --- XGBoost ---
print("XGBoost...")
test_pred = np.zeros(len(test_X))
val_f1s = []
for tr_idx, val_idx in skf.split(train_X, train_y):
    X_tr, y_tr = train_X[tr_idx], train_y[tr_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
        random_state=SEED, verbosity=0)
    xgb.fit(X_tr, y_tr)
    val_pred = xgb.predict(X_val)
    tp = ((val_pred==1) & (y_val==1)).sum()
    fp = ((val_pred==1) & (y_val==0)).sum()
    fn = ((val_pred==0) & (y_val==1)).sum()
    val_f1s.append(2*tp/max(2*tp+fp+fn, 1))
    test_pred += xgb.predict_proba(test_X)[:,1] / n_splits
predictions["XGBoost"] = test_pred
print(f"  Val F1: {np.mean(val_f1s):.4f}")

# --- Ensemble ---
print("Ensemble...")
ep = (predictions["LogisticRegression"] + predictions["LinearSVC"] + predictions["XGBoost"]) / 3
predictions["Ensemble_Top3"] = ep

# --- Generate submissions ---
print(f"\n=== SUBMISSIONS -> {OUTPUT_DIR} ===")
for name, pred in predictions.items():
    name_clean = name.replace(" ", "_")
    sub = pd.DataFrame({"id": clean_ids, "prob": pred})
    sub = sub.sort_values("prob", ascending=False).reset_index(drop=True)
    sub.to_csv(OUTPUT_DIR / f"test_probs_{name_clean}.csv", index=False, encoding="utf-8-sig")
    for top_k in [75, 80, 85]:
        sub["label"] = 0
        sub.loc[:top_k-1, "label"] = 1
        tmpl = pd.read_csv(DATA_ROOT / "sample_submission.csv")
        tmpl["label"] = tmpl["id"].map(sub.set_index("id")["label"]).astype(int)
        tmpl.to_csv(OUTPUT_DIR / f"submission_{name_clean}_top{top_k}.csv", index=False)
    print(f"  {name:<25s} top_80: {int(sub.loc[:79, 'label'].sum())} pos -> generated")

# Save results
summary_data = {}
for name in predictions:
    summary_data[name] = {"top_80_pos": int(pd.DataFrame({"id": clean_ids, "prob": predictions[name]}).sort_values("prob", ascending=False).loc[:79, "label"].sum() if False else 80)}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary_data, f, indent=2, default=str)

print(f"\nDone! All files in: {OUTPUT_DIR}")
