"""
Meteor-Mamba 批量陨石识别推理脚本
使用 Meteor-Mamba 多模态模型对测试集图片进行零样本分类
判断每张图片是否为陨石 (meteorite: 1 / non-meteorite: 0)

用法: python meteor_batch_infer.py
输出: submission_meteor.csv
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
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

# ==========================================
# 配置
# ==========================================
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images" / "test_images"
TEMPLATE_CSV = Path(__file__).parent.parent / "sample_submission.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_CSV = OUTPUT_DIR / "submission_meteor.csv"
OUTPUT_JSON = OUTPUT_DIR / "meteor_results.json"

# 推理参数
IMAGE_SIZE = 490  # Meteor-Mamba 使用的图像尺寸
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# 提示词 - 让模型判断是否为陨石
QUESTION = (
    "Is this image showing a meteorite (a rock from outer space that fell to Earth)? "
    "Look for features like dark fusion crust, rounded shape, regmaglypts (thumbprint-like indentations), "
    "metallic appearance, or charred surface. "
    "Answer only YES if it is a meteorite, or NO if it is not a meteorite."
)

# 批量处理参数
BATCH_SAVE_INTERVAL = 20  # 每处理多少张图片保存一次中间结果


def load_models():
    """加载 Meteor-Mamba 和 Meteor-MLM 模型"""
    from meteor.load_mmamba import load_mmamba
    from meteor.load_meteor import load_meteor

    print("Loading Meteor-Mamba (Mamba-130M)...")
    mmamba = load_mmamba('BK-Lee/Meteor-Mamba').cuda()

    print("Loading Meteor-MLM (InternLM2-7B, 4-bit)...")
    meteor, tok_meteor = load_meteor('BK-Lee/Meteor-MLM', bits=4)

    # 冻结模型
    from utils.utils import freeze_model
    freeze_model(mmamba)
    freeze_model(meteor)

    print("Models loaded successfully!")
    return mmamba, meteor, tok_meteor


def process_single_image(mmamba, meteor, tok_meteor, image_path, device, img_token_number):
    """对单张图片进行陨石判断，返回 (predicted_label, confidence, raw_response)"""
    try:
        # 加载并预处理图像
        image = F.interpolate(
            pil_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bicubic'
        ).squeeze(0)

        inputs = [{'image': image, 'question': QUESTION}]

        with torch.inference_mode():
            # Meteor Mamba - 提取 trajectory of rationale 特征
            mmamba_inputs = mmamba.eval_process(
                inputs=inputs, tokenizer=tok_meteor, device=device,
                img_token_number=img_token_number
            )
            clip_features = meteor.clip_features(mmamba_inputs['image'])
            mmamba_inputs.update({"image_features": clip_features})
            mmamba_outputs = mmamba(**mmamba_inputs)

            # Meteor MLM - 生成回答
            meteor_inputs = meteor.eval_process(
                inputs=inputs, data='demo', tokenizer=tok_meteor,
                device=device, img_token_number=img_token_number
            )
            meteor_inputs.update({"image_features": clip_features})
            meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})

            generate_ids = meteor.generate(
                **meteor_inputs, do_sample=DO_SAMPLE,
                max_new_tokens=MAX_NEW_TOKENS, top_p=TOP_P,
                temperature=TEMPERATURE, use_cache=True
            )

        # 解码文本
        decoded_text = tok_meteor.batch_decode(
            generate_ids, skip_special_tokens=True
        )[0].split('assistant\n')[-1].split('[U')[0].strip().lower()

        # 解析结果
        if 'yes' in decoded_text:
            return 1, decoded_text
        elif 'no' in decoded_text:
            return 0, decoded_text
        else:
            # 如果无法判断，检查是否包含陨石相关词汇
            meteorite_keywords = ['meteorite', 'crust', 'regmaglypt', 'fusion', 'space rock']
            if any(kw in decoded_text for kw in meteorite_keywords):
                return 1, decoded_text
            return 0, decoded_text

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0, f"ERROR: {str(e)}"


def main():
    start_time = time.time()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查设备
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(f"Device: {'cuda:' + str(device) if torch.cuda.is_available() else 'cpu'}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")

    # 加载模型
    mmamba, meteor, tok_meteor = load_models()

    # 计算图像 token 数量 (与 demo 代码一致)
    image_token_number = int((490 / 14) ** 2)

    # 读取测试图片列表
    image_files = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} test images in {TEST_IMAGES_DIR}")

    # 读取模板以确保顺序正确
    template_df = pd.read_csv(TEMPLATE_CSV)
    print(f"Template CSV: {len(template_df)} entries")

    # 批量推理
    results = {}
    intermediate_file = OUTPUT_DIR / "intermediate_results.json"

    # 尝试加载中间结果以支持断点续传
    if intermediate_file.exists():
        with open(intermediate_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} intermediate results, resuming...")

    processed = 0
    pbar = tqdm(enumerate(image_files), total=len(image_files), desc="Classifying meteorites")

    for i, img_path in pbar:
        img_name = img_path.name

        if img_name in results:
            processed += 1
            continue

        label, response = process_single_image(
            mmamba, meteor, tok_meteor, image_path=img_path,
            device=device, img_token_number=image_token_number
        )

        results[img_name] = {
            "label": label,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        processed += 1

        # 更新进度条
        meteorite_count = sum(1 for r in results.values() if r["label"] == 1)
        pbar.set_postfix({
            "meteorites": meteorite_count,
            "total": processed
        })

        # 定期保存中间结果
        if processed % BATCH_SAVE_INTERVAL == 0:
            with open(intermediate_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 最终保存
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成提交文件
    submission_df = template_df.copy()
    submission_df["label"] = submission_df["id"].map(
        lambda x: results.get(x, {}).get("label", 0)
    )
    submission_df["label"] = submission_df["label"].astype(int)
    submission_df.to_csv(OUTPUT_CSV, index=False)

    # 统计结果
    n_meteorites = int(submission_df["label"].sum())
    n_non = int((submission_df["label"] == 0).sum())

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total images:       {len(image_files)}")
    print(f"Meteorites (1):     {n_meteorites}")
    print(f"Non-meteorites (0): {n_non}")
    print(f"Time elapsed:       {elapsed/60:.1f} minutes")
    print(f"Avg per image:      {elapsed/len(image_files):.2f} seconds")
    print(f"\nSubmission saved to: {OUTPUT_CSV}")
    print(f"Detailed results:   {OUTPUT_JSON}")

    # 打印所有预测为陨石的图片
    meteorite_images = [
        name for name, info in results.items() if info["label"] == 1
    ]
    print(f"\nMeteorite images ({len(meteorite_images)}):")
    for name in sorted(meteorite_images):
        resp = results[name]["response"]
        print(f"  {name}: {resp[:80]}...")

    return results, submission_df


if __name__ == "__main__":
    main()
