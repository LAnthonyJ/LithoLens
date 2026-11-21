"""Copy checkpoints and run inference test"""
import subprocess, os

test_dir = "/data/数据科学实践project/repro_test"

# Copy checkpoints
copies = [
    ("/data/数据科学实践project/runs/final_top86/best_head.pth",
     f"{test_dir}/checkpoints/dinov3_tex_mlp_head.pth"),
    ("/data/数据科学实践project/runs/v13_proper/convnextv2_base_frozen_head_v2/test_prob.csv",
     f"{test_dir}/checkpoints/convnextv2_base_test_prob.csv"),
    ("/data/数据科学实践project/runs/final_top86/texture_cache.npz",
     f"{test_dir}/checkpoints/texture_cache.npz"),
]
for src, dst in copies:
    r = subprocess.run(["cp", src, dst], capture_output=True, text=True)
    print(f"cp {src.split('/')[-1]}: {'OK' if r.returncode == 0 else 'FAIL'}")

# Check
import os as _os
for f in _os.listdir(f"{test_dir}/checkpoints"):
    print(f"  {f}")

# Run inference
print("\nRunning inference...")
r = subprocess.run([
    "python", "run.py",
    "--data_path", "/data/数据科学实践project",
    "--checkpoint_dir", "./checkpoints",
    "--dino_weights", "/data/models/dinov3_large/model.safetensors",
    "--output", "submission_verified.csv",
], cwd=test_dir, capture_output=True, text=True, timeout=600)
print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[-500:])

# Compare
import pandas as pd
test = pd.read_csv(f"{test_dir}/submission_verified.csv")
best = pd.read_csv("/data/数据科学实践project/runs/final_top86/submission_blend_top86.csv")
d = (test["label"] != best["label"]).sum()
print(f"\nDifferences: {d}/194 | Test: {test['label'].sum()} pos | Best: {best['label'].sum()} pos")
print("MATCH - Reproducible!" if d == 0 else f"MISMATCH: {d} differ")
