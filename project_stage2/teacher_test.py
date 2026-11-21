"""Simulate teacher verification: clone repo + download cloud archive + run inference"""
import subprocess, os, sys

ROOT = "/data/teacher_test"
DATA = "/data/数据科学实践project"
DINO = "/data/models/dinov3_large/model.safetensors"

# Step 1: Setup clean directory (simulating git clone + unzip)
os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)

# Copy repo files (simulating git clone - files already uploaded to ROOT)
# run.py and requirements.txt are placed by the test harness
print("Repo files: run.py, requirements.txt (from GitHub)")

# Copy checkpoints (simulating unzip from cloud)
checkpoints = {
    "best_head.pth": "dinov3_tex_mlp_head.pth",
    "test_prob.csv": "convnextv2_base_test_prob.csv",
    "texture_cache.npz": "texture_cache.npz",
}
sources = {
    "best_head.pth": f"{DATA}/runs/final_top86/best_head.pth",
    "test_prob.csv": f"{DATA}/runs/v13_proper/convnextv2_base_frozen_head_v2/test_prob.csv",
    "texture_cache.npz": f"{DATA}/runs/final_top86/texture_cache.npz",
}
for src_name, dst_name in checkpoints.items():
    subprocess.run(["cp", sources[src_name], f"{ROOT}/checkpoints/{dst_name}"], check=True)

print("Setup complete: repo files + checkpoints in place")
print(f"  {ROOT}/run.py")
print(f"  {ROOT}/checkpoints/ ({len(os.listdir(f'{ROOT}/checkpoints'))} files)")

# Step 2: Run inference EXACTLY as README instructs
print("\n=== Running: python run.py --data_path ./data --checkpoint_dir ./checkpoints ... ===\n")
result = subprocess.run([
    sys.executable, "run.py",
    "--data_path", DATA,
    "--checkpoint_dir", "./checkpoints",
    "--dino_weights", DINO,
    "--output", "submission_blend_top86.csv",
], cwd=ROOT, capture_output=True, text=True, timeout=600)

# Print filtered output (no tqdm bars)
for line in result.stdout.split("\n"):
    if line.strip() and "it/s" not in line and "%" not in line:
        print(line)
if result.stderr:
    for line in result.stderr.split("\n"):
        if line.strip() and "it/s" not in line and "%" not in line:
            print(f"  [stderr] {line}")

# Step 3: Verify output
import pandas as pd
output_csv = f"{ROOT}/submission_blend_top86.csv"
df = pd.read_csv(output_csv)

print(f"\n=== Verification ===")
print(f"File: submission_blend_top86.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Positives (label=1): {df['label'].sum()}")
print(f"Negatives (label=0): {(df['label'] == 0).sum()}")

# Check against known best
best = pd.read_csv(f"{DATA}/runs/final_top86/submission_blend_top86.csv")
diffs = (df["label"] != best["label"]).sum()

print(f"\nComparison with best known submission:")
print(f"  Differences: {diffs}/194")
if diffs == 0:
    print("  RESULT: MATCH! Teacher can reproduce exact same submission.")
else:
    print(f"  RESULT: MISMATCH - {diffs} samples differ")
    sys.exit(1)
