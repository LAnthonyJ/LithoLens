"""
Meteor-Mamba LoRA 微调 - 陨石分类
在训练集上使用 LoRA 微调 Meteor-MLM，冻结 CLIP 和 Meteor-Mamba
"""
import os
import sys
import json
import time
from pathlib import Path

# Setup
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
METEOR_PATH = '/tmp/Meteor'
sys.path.insert(0, METEOR_PATH)

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# Config
# ==========================================
DATA_ROOT = '/data/数据科学实践project'
TRAIN_CSV = os.path.join(DATA_ROOT, 'train_labels.csv')
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train_images', 'train_images')
MODEL_DIR = '/data/models'
OUTPUT_DIR = '/data/数据科学实践project/meteor_output/lora_checkpoints'

IMAGE_SIZE = 490
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION = 8
WARMUP_STEPS = 50
MAX_STEPS = 1000
SAVE_STEPS = 200

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


class MeteoriteDataset(Dataset):
    """陨石分类数据集，格式化为 VQA"""
    def __init__(self, csv_path, img_dir, tok_meteor, image_size=490):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tok_meteor = tok_meteor
        self.image_size = image_size

        # 将 label 转为文字
        self.label_to_text = {1: 'yes', 0: 'no'}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id'])
        label = int(row['label'])

        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        image_tensor = F.interpolate(
            pil_to_tensor(image).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bicubic'
        ).squeeze(0)

        question = "Is this a meteorite? Answer yes or no."
        answer = self.label_to_text[label]

        return {
            'image': image_tensor,
            'question': question,
            'answer': answer,
            'label': label,
            'img_id': row['id']
        }


def load_models():
    """加载模型并应用 LoRA"""
    from meteor.load_mmamba import load_mmamba
    from meteor.load_meteor import load_meteor
    from utils.utils import freeze_model

    print("Loading Meteor-Mamba...")
    mmamba = load_mmamba(os.path.join(MODEL_DIR, 'meteor_mamba')).cuda()
    freeze_model(mmamba)

    print("Loading Meteor-MLM...")
    meteor, tok_meteor = load_meteor(os.path.join(MODEL_DIR, 'meteor_mlm'), bits=0)
    freeze_model(meteor)

    # Apply LoRA to Meteor-MLM's language model
    print("Applying LoRA to Meteor-MLM...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Unfreeze the model for LoRA application
    for param in meteor.parameters():
        param.requires_grad = False

    meteor = get_peft_model(meteor, lora_config)
    meteor.print_trainable_parameters()

    # Freeze vision tower and Mamba
    freeze_model(mmamba)

    return mmamba, meteor, tok_meteor


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    mmamba, meteor, tok_meteor = load_models()
    image_token_number = int((IMAGE_SIZE / 14) ** 2)

    # Dataset
    dataset = MeteoriteDataset(TRAIN_CSV, TRAIN_IMG_DIR, tok_meteor, IMAGE_SIZE)

    # Split train/val
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, stratify=dataset.df['label'], random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in meteor.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=min(total_steps, MAX_STEPS),
    )

    # Training loop
    meteor.train()
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")

        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            answers = batch['answer']

            # Prepare inputs through Meteor pipeline
            inputs = [{'image': images[i], 'question': batch['question'][i]} for i in range(len(images))]

            with torch.no_grad():
                # Meteor Mamba ToR features
                mmamba_inputs = mmamba.eval_process(
                    inputs=inputs, tokenizer=tok_meteor, device=device,
                    img_token_number=image_token_number
                )
                clip_features = meteor.clip_features(mmamba_inputs['image'])
                mmamba_inputs.update({"image_features": clip_features})
                mmamba_outputs = mmamba(**mmamba_inputs)

            # Meteor MLM with labels
            meteor_inputs = meteor.eval_process(
                inputs=inputs, data='demo', tokenizer=tok_meteor,
                device=device, img_token_number=image_token_number
            )
            meteor_inputs.update({"image_features": clip_features})
            meteor_inputs.update({"tor_features": mmamba_outputs.tor_features})

            # Add labels for training
            answer_tokens = tok_meteor(
                answers, return_tensors='pt', padding=True, add_special_tokens=False
            ).input_ids.to(device)

            # Create labels: -100 for non-answer tokens
            labels = torch.full_like(meteor_inputs['input_ids'], -100)
            for i, ans_tok in enumerate(answer_tokens):
                # Put answer at the end of the sequence
                seq_len = meteor_inputs['input_ids'].shape[1]
                ans_len = len(ans_tok)
                if seq_len >= ans_len:
                    labels[i, seq_len - ans_len:seq_len] = ans_tok

            meteor_inputs['labels'] = labels

            # Forward pass with loss
            outputs = meteor(**meteor_inputs)
            loss = outputs.loss / GRADIENT_ACCUMULATION

            loss.backward()
            total_loss += loss.item()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in meteor.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': global_step,
            })

            if global_step >= MAX_STEPS:
                break

            # Validation
            if global_step > 0 and global_step % SAVE_STEPS == 0:
                val_acc = validate(meteor, mmamba, tok_meteor, val_loader, device, image_token_number)
                print(f"\nStep {global_step}: Val Acc = {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    meteor.save_pretrained(os.path.join(OUTPUT_DIR, 'best_lora'))
                    print(f"Saved best model (acc={best_val_acc:.4f})")

                meteor.train()

        if global_step >= MAX_STEPS:
            break

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Final save
    meteor.save_pretrained(os.path.join(OUTPUT_DIR, 'final_lora'))
    print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")


@torch.no_grad()
def validate(meteor, mmamba, tok_meteor, val_loader, device, img_token_number):
    meteor.eval()
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        images = batch['image'].to(device)
        labels = batch['label']

        inputs = [{'image': images[i], 'question': batch['question'][i]} for i in range(len(images))]

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

        # Generate
        generate_ids = meteor.generate(
            **meteor_inputs, do_sample=False,
            max_new_tokens=8, use_cache=True
        )

        # Decode and compare
        for i, (ids, label) in enumerate(zip(generate_ids, labels)):
            decoded = tok_meteor.decode(ids, skip_special_tokens=True).lower()
            pred = 1 if 'yes' in decoded else 0
            if pred == int(label):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    train()
