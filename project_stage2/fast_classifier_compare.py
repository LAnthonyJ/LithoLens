"""
Compare classifiers using cached features
"""
from pathlib import Path
import sys, json, pickle, random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_ROOT = Path("/data/数据科学实践project")
FEATURES_DIR = DATA_ROOT / "runs" / "dinov2_features"
OUTPUT_DIR = DATA_ROOT / "runs" / f"classifier_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

print(f"OUTPUT: {OUTPUT_DIR}")
sys.stdout.flush()

# ==========================================
# Load cached features
# ==========================================
print("Load cached features...")
train_data = np.load(FEATURES_DIR / "train_features.npz")
test_data = np.load(FEATURES_DIR / "test_features.npz")
train_X = train_data['X']
train_y = train_data['y'].astype(int)
test_X = test_data['X']
test_ids = list(test_data['ids'])
print(f"Train: {train_X.shape}, Test: {test_X.shape}")
print(f"Labels: {np.bincount(train_y)}")
sys.stdout.flush()

# ==========================================
# Define classifiers (excluding SVM probability=True)
# ==========================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight='balanced', dual=False, max_iter=2000, random_state=SEED)),
    "kNN_k25_cosine": KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine'),
    "RandomForest_300": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=(train_y==0).sum()/max((train_y==1).sum(),1), random_state=SEED, verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=SEED, verbose=-1),
    "LightGBM_RF": LGBMClassifier(boosting_type='rf', n_estimators=300, num_leaves=64, class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=-1),
}

results = []

print("\n" + "=" * 60)
print(f"{'Classifier':<25s} {'Val Acc':>10s} {'Val F1':>10s} {'Time':>8s}")
print("-" * 60)
sys.stdout.flush()

import time

for name, clf in classifiers.items():
    t0 = time.time()
    val_accs, val_f1s = [], []
    test_pred = np.zeros(len(test_X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        X_tr, y_tr = train_X[train_idx], train_y[train_idx]
        X_val, y_val = train_X[val_idx], train_y[val_idx]

        clf.fit(X_tr, y_tr)
        val_pred = clf.predict(X_val)
        val_acc = (val_pred == y_val).mean()
        tp = ((val_pred == 1) & (y_val == 1)).sum()
        fp = ((val_pred == 1) & (y_val == 0)).sum()
        fn = ((val_pred == 0) & (y_val == 1)).sum()
        val_f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # Accumulate test probabilities
        if hasattr(clf, 'predict_proba'):
            test_pred += clf.predict_proba(test_X)[:, 1] / n_splits
        else:
            # LinearSVC Without predict_proba, use decision_function
            d = clf.decision_function(test_X)
            test_pred += (1 / (1 + np.exp(-d))) / n_splits

    elapsed = time.time() - t0
    mean_acc = np.mean(val_accs)
    mean_f1 = np.mean(val_f1s)

    results.append({
        'name': name,
        'val_acc': mean_acc,
        'val_f1': mean_f1,
        'test_pred': test_pred.copy(),
        'time': elapsed,
    })

    print(f"{name:<25s} {mean_acc:.4f}±{np.std(val_accs):.4f}  {mean_f1:.4f}±{np.std(val_f1s):.4f}  {elapsed:.0f}s")
    sys.stdout.flush()

# ==========================================
# Sort results
# ==========================================
print("\n" + "=" * 60)
print("Ranking (by Val F1)")
print("=" * 60)
results.sort(key=lambda r: r['val_f1'], reverse=True)
for i, r in enumerate(results):
    print(f"{i+1}. {r['name']:<25s} F1={r['val_f1']:.4f}  Acc={r['val_acc']:.4f}")
sys.stdout.flush()

# ==========================================
# Write submissions for the top three
# ==========================================
print("\n" + "=" * 60)
print("Write the submission (top_k = 75, 80, 85)")
print("=" * 60)

clean_ids = [Path(p).name for p in test_ids]

for r in results[:3]:
    name = r['name'].replace(' ', '_')
    sub_df = pd.DataFrame({'id': clean_ids, 'prob': r['test_pred']})
    sub_df = sub_df.sort_values('prob', ascending=False).reset_index(drop=True)

    for top_k in [75, 80, 85]:
        sub_df['label'] = 0
        sub_df.loc[:top_k - 1, 'label'] = 1
        template = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
        template['label'] = template['id'].map(sub_df.set_index('id')['label']).astype(int)
        csv_path = OUTPUT_DIR / f'submission_{name}_top{top_k}.csv'
        template.to_csv(csv_path, index=False)
        print(f"  {name} top_{top_k}: {int(sub_df['label'].sum())} pos -> {csv_path.name}")
    sys.stdout.flush()

# ==========================================
# ensemble Top-3
# ==========================================
print("\n--- ensemble Top-3 ---")
ensemble_pred = np.zeros(len(test_X))
for r in results[:3]:
    ensemble_pred += r['test_pred'] / 3

sub_df = pd.DataFrame({'id': clean_ids, 'prob': ensemble_pred})
sub_df = sub_df.sort_values('prob', ascending=False).reset_index(drop=True)
sub_df.to_csv(OUTPUT_DIR / 'test_probs_ensemble_top3.csv', index=False, encoding='utf-8-sig')

for top_k in [75, 80, 85]:
    sub_df['label'] = 0
    sub_df.loc[:top_k - 1, 'label'] = 1
    template = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    template['label'] = template['id'].map(sub_df.set_index('id')['label']).astype(int)
    csv_path = OUTPUT_DIR / f'submission_ensemble_top3_top{top_k}.csv'
    template.to_csv(csv_path, index=False)
    print(f"  ensemble top_{top_k}: {int(sub_df['label'].sum())} pos -> {csv_path.name}")
sys.stdout.flush()

# Save summary
summary = {r['name']: {'val_f1': r['val_f1'], 'val_acc': r['val_acc'], 'time_s': r['time']} for r in results}
with open(OUTPUT_DIR / 'classifier_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n🎉 Finished. Output: {OUTPUT_DIR}")
