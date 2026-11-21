"""
Meteor-Mamba 陨石批量识别 - 服务器版
在 Linux GPU 服务器上运行，对测试集进行零样本分类
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add Meteor to path
METEOR_PATH = '/tmp/Meteor'
sys.path.insert(0, METEOR_PATH)

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

# ==========================================
# Server paths
# ==========================================
TEST_IMAGES_DIR = '/data/数据科学实践project/test_images/test_images'
TEMPLATE_CSV = '/data/数据科学实践project/sample_submission.csv'
OUTPUT_DIR = '/data/数据科学实践project/meteor_output'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'submission_meteor.csv')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'meteor_results.json')

# ==========================================
# Inference parameters
# ==========================================
IMAGE_SIZE = 490
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.9

QUESTION = (
    "Is this image showing a meteorite (a rock from outer space that fell to Earth)? "
    "Look for features like dark fusion crust, rounded shape, regmaglypts (thumbprint-like indentations), "
    "metallic appearance, or charred surface. "
    "Answer only YES if it is a meteorite, or NO if it is not a meteorite."
)


def load_models():
    """Load Meteor-Mamba and Meteor-MLM"""
    from meteor.load_mmamba import load_mmamba
    from meteor.load_meteor import load_meteor
    from utils.utils import freeze_model

    print("Loading Meteor-Mamba (Mamba-130M)...")
    mmamba = load_mmamba('BK-Lee/Meteor-Mamba').cuda()

    print("Loading Meteor-MLM (InternLM2-7B, 4-bit)...")
    meteor, tok_meteor = load_meteor('BK-Lee/Meteor-MLM', bits=4)

    freeze_model(mmamba)
    freeze_model(meteor)

    print("Models loaded!")
    return mmamba, meteor, tok_meteor


def classify_image(mmamba, meteor, tok_meteor, image_path, device, img_token_number):
    """Classify a single image as meteorite (1) or non-meteorite (0)"""
    try:
        # Load and preprocess image
        image = F.interpolate(
            pil_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bicubic'
        ).squeeze(0)

        inputs = [{'image': image, 'question': QUESTION}]

        with torch.inference_mode():
            # Meteor Mamba
            mmamba_inputs = mmamba.eval_process(
                inputs=inputs, tokenizer=tok_meteor, device=device,
                img_token_number=img_token_number
            )
            clip_features = meteor.clip_features(mmamba_inputs['image'])
            mmamba_inputs.update({"image_features": clip_features})
            mmamba_outputs = mmamba(**mmamba_inputs)

            # Meteor MLM
            meteor_inputs = meteor.eval_process(
                inputs=inputs, data='demo', tokenizer=tok_meteor,
                device=device, img_token_number=img_token_number
            )
            meteor_inputs.update({"image_features": clip_features})
            meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})

            generate_ids = meteor.generate(
                **meteor_inputs, do_sample=True,
                max_new_tokens=MAX_NEW_TOKENS, top_p=TOP_P,
                temperature=TEMPERATURE, use_cache=True
            )

        # Decode text
        decoded = tok_meteor.batch_decode(
            generate_ids, skip_special_tokens=True
        )[0].split('assistant\n')[-1].split('[U')[0].strip().lower()

        if 'yes' in decoded:
            return 1, decoded
        elif 'no' in decoded:
            return 0, decoded
        else:
            keywords = ['meteorite', 'crust', 'regmaglypt', 'fusion', 'space rock']
            if any(kw in decoded for kw in keywords):
                return 1, decoded
            return 0, decoded

    except Exception as e:
        print(f"Error on {image_path}: {e}")
        return 0, f"ERROR: {str(e)}"


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load models
    mmamba, meteor, tok_meteor = load_models()
    image_token_number = int((490 / 14) ** 2)

    # Get image list
    image_files = sorted(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    print(f"\nFound {len(image_files)} test images")

    template_df = pd.read_csv(TEMPLATE_CSV)

    # Load existing results for resume capability
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

        label, response = classify_image(
            mmamba, meteor, tok_meteor,
            image_path=str(img_path),
            device=device,
            img_token_number=image_token_number
        )

        results[img_name] = {
            "label": label,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        n_met = sum(1 for r in results.values() if r["label"] == 1)
        pbar.set_postfix({"meteorites": n_met})

        # Save every 20 images
        if len(results) % 20 == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate submission
    submission_df = template_df.copy()
    submission_df["label"] = submission_df["id"].map(
        lambda x: results.get(x, {}).get("label", 0)
    )
    submission_df["label"] = submission_df["label"].astype(int)
    submission_df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    n_meteorites = int(submission_df["label"].sum())
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"Total: {len(image_files)} | Meteorites: {n_meteorites} | Non: {len(image_files) - n_meteorites}")
    print(f"Time: {elapsed/60:.1f} min | Avg: {elapsed/len(image_files):.1f}s per image")
    print(f"Submission: {OUTPUT_CSV}")

    # List meteorites
    meteorite_imgs = sorted([n for n, r in results.items() if r["label"] == 1])
    print(f"\nMeteorite images ({len(meteorite_imgs)}):")
    for name in meteorite_imgs:
        print(f"  {name}")


if __name__ == "__main__":
    main()
