"""
Qwen3.5 zero-shot meteorite classification
Classify test images with the Qwen3.5 model installed on the server
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# Server paths
# ==========================================
MODEL_PATH = '/data/qwen_model/Qwen/Qwen3___5-0___8B'
TEST_IMAGES_DIR = '/data/数据科学实践project/test_images/test_images'
TEMPLATE_CSV = '/data/数据科学实践project/sample_submission.csv'
OUTPUT_DIR = '/data/数据科学实践project/meteor_output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'submission_qwen.csv')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'qwen_results.json')

# ==========================================
# Prompt for meteorite classification
# ==========================================
QUESTION = (
    "Is this image showing a meteorite (a rock from outer space)? "
    "Look for dark fusion crust, rounded shape, regmaglypts (thumbprint-like indentations), "
    "metallic appearance, or charred surface. "
    "Answer ONLY with YES or NO."
)


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model and processor
    print(f"Loading Qwen3.5 from {MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    print("Model loaded!")

    # Get image list
    image_files = sorted(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    print(f"\nFound {len(image_files)} test images")

    template_df = pd.read_csv(TEMPLATE_CSV)

    # Resume capability
    results = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            results = json.load(f)
        print(f"Resumed {len(results)} previous results")

    # Run inference
    pbar = tqdm(image_files, desc="Classifying")
    for img_path in pbar:
        img_name = img_path.name
        if img_name in results:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QUESTION},
                ]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            response = processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()

            if 'yes' in response:
                label = 1
            elif 'no' in response:
                label = 0
            else:
                keywords = ['meteorite', 'crust', 'regmaglypt', 'fusion', 'space rock', 'appears to be']
                label = 1 if any(kw in response for kw in keywords) else 0

        except Exception as e:
            print(f"Error on {img_name}: {e}")
            response = f"ERROR: {str(e)[:100]}"
            label = 0

        results[img_name] = {"label": label, "response": response}
        n_met = sum(1 for r in results.values() if r["label"] == 1)
        pbar.set_postfix({"meteorites": n_met})

        # Save periodically
        if len(results) % 20 == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate submission
    submission_df = template_df.copy()
    submission_df["label"] = submission_df["id"].map(lambda x: results.get(x, {}).get("label", 0))
    submission_df["label"] = submission_df["label"].astype(int)
    submission_df.to_csv(OUTPUT_CSV, index=False)

    n_meteorites = int(submission_df["label"].sum())
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"Total: {len(image_files)} | Meteorites: {n_meteorites} | Non: {len(image_files) - n_meteorites}")
    print(f"Time: {elapsed/60:.1f} min | Avg: {elapsed/len(image_files):.1f}s per image")
    print(f"Submission: {OUTPUT_CSV}")

    meteorite_imgs = sorted([n for n, r in results.items() if r["label"] == 1])
    print(f"\nMeteorite images ({len(meteorite_imgs)}):")
    for name in meteorite_imgs:
        print(f"  {name}")


if __name__ == "__main__":
    main()
