import numpy as np
import json
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score
from data_loader.utils import build_dataset
from loss.focal_loss import FocalLoss
from loss.utils import get_criterion
from models.utils import build_model, save_model, load_model
from utils import inference_and_save
def find_best_threshold(all_labels, all_probs):
    labels = np.asarray(all_labels, dtype=np.int64)
    probs = np.asarray(all_probs, dtype=np.float32)

    if labels.size == 0:
        return 0.5, 0.0

    candidate_thresholds = np.arange(0.30, 0.801, 0.05)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        preds = (probs >= threshold).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)

        if score > best_f1 or (score == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1
def get_threshold_metadata_path(model_path):
    base_path, _ = os.path.splitext(model_path)
    return f"{base_path}.json"
def save_threshold_metadata(model_path, threshold, best_f1, epoch):
    metadata_path = get_threshold_metadata_path(model_path)
    metadata = {
        "best_threshold": float(threshold),
        "best_f1": float(best_f1),
        "best_epoch": int(epoch),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
def load_threshold_metadata(model_path, default_threshold=0.5):
    metadata_path = get_threshold_metadata_path(model_path)

    if not os.path.exists(metadata_path):
        return default_threshold

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return float(metadata.get("best_threshold", default_threshold))
if __name__ == '__main__':
    print("Hello world!")
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--data_path", default="./dataset")
    parser.add_argument('--model', '-m', type=str, default='efficientnet_b3', help='Choose architecture.')
    parser.add_argument('--loss', type=str, default='ce')

    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train. Default to 0 means no training')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Train Batch size.')
    parser.add_argument('--test_bs', type=int, default=512, help="Test Batch size")

    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4, help='The initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (not heavily used in AdamW).')
    parser.add_argument('--decay', '-d', type=float, default=1e-3, help='Weight decay (L2 penalty).')
    
    parser.add_argument('--non_meteorite_weight', type=float, default=2.0, 
                        help='非陨石类别的损失权重 (陨石固定为 1.0)。因为测试集中非陨石比例比训练集大很多，增加该值会让模型更关注非陨石样本，提升模型对非陨石的识别能力。 4.45 may be better')

    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads for dataloader.')

    parser.add_argument("--load_model_path", default=None, type=str, help="Set to the model path that you want to load")
    parser.add_argument('--save', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--seed', type=int, default=35)

    args = parser.parse_args()

    state = {k: v for k, v in args._get_kwargs()}
    print(f"Hyperparameters for the experiment")
    print(state)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- 1. 数据集准备与自定义划分 ----------------
    if args.epochs > 0:
        full_train_data, num_classes = build_dataset(args.data_path, "train")
        
        # 获取底层 DataFrame
        df = full_train_data.df
        
        # 分离出非陨石(class 0)和陨石(class 1)的原始索引
        class0_indices = df[df['label'] == 0].index.values
        class1_indices = df[df['label'] == 1].index.values

        # 打乱索引以保证随机性
        np.random.shuffle(class0_indices)
        np.random.shuffle(class1_indices)

        # 设定验证集占总训练集数据的 10%
        val_total = int(0.1 * len(full_train_data))
        
        # 强制验证集比例为 非陨石 : 陨石 = 4 : 1
        val_class1_count = val_total // 5
        val_class0_count = val_class1_count * 4

        # 划分验证集索引
        val_idx = np.concatenate([
            class0_indices[:val_class0_count], 
            class1_indices[:val_class1_count]
        ])
        
        # 剩余的全部作为训练集
        train_idx = np.concatenate([
            class0_indices[val_class0_count:], 
            class1_indices[val_class1_count:]
        ])

        train_data = Subset(full_train_data, train_idx)
        val_data = Subset(full_train_data, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers)
        
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
            
        print(f"\n[Data Split] Training Set: {len(train_data)} | Validation Set: {len(val_data)}")
        print(f"[Validation Ratio] Non-Meteorite: {val_class0_count}, Meteorite: {val_class1_count} (Ratio = 4:1)\n")

    test_data, num_classes = build_dataset(args.data_path, "test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_bs, shuffle=False,
        num_workers=args.num_workers)

    # ---------------- 2. 构建与加载模型 ----------------
    net = build_model(args, device=device, num_classes=num_classes)

    if args.load_model_path is not None:
        print(f"\n[Info] Loading weights from {args.load_model_path}...")
        net = load_model(net, args.load_model_path, device=device)

    best_threshold = 0.5

    # ---------------- 3. 训练与验证逻辑 ----------------
    if args.epochs > 0:
        print("----------------Training & Validation--------------------\n")

        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # ---------- 修改处：非陨石为 args.non_meteorite_weight，陨石为 1.0 ----------
        if args.loss == 'ce':
            class_weights = torch.FloatTensor([args.non_meteorite_weight, 1.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"[Loss Info] Using CrossEntropy with Class Weights: [Non-Meteorite: {args.non_meteorite_weight}, Meteorite: 1.0]\n")
        elif args.loss == "focal":
            criterion = FocalLoss()
        else:
            criterion = get_criterion(args)

        best_f1 = 0.0
        best_model_state = None
        os.makedirs("./checkpoints", exist_ok=True)
        best_model_path = f"./checkpoints/{args.model}_{args.loss}_best_f1.pth"

        for e in range(1, args.epochs + 1):
            net.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {e}/{args.epochs} [Train]")
            
            total_train_loss = 0.0

            for images, labels in train_bar:
                images = images.to(device)
                labels = labels.to(device)

                logits = net(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)

            net.eval()
            val_loss = 0.0
            total = 0
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    logits = net(images)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    probabilities = torch.softmax(logits, dim=1)[:, 1]

                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probabilities.cpu().numpy())
            
            avg_val_loss = val_loss / total
            epoch_best_threshold, val_f1 = find_best_threshold(all_labels, all_probs)
            val_preds = (np.asarray(all_probs) >= epoch_best_threshold).astype(np.int64)
            val_acc = 100. * (val_preds == np.asarray(all_labels)).mean()
            
            print(f"-> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Best Threshold: {epoch_best_threshold:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = float(epoch_best_threshold)
                best_model_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                if args.save == "True":
                    print(f"🌟 Found new best Val F1: {best_f1:.4f}! Saving model to {best_model_path}...")
                    save_model(net, save_path=best_model_path)
                    save_threshold_metadata(best_model_path, best_threshold, best_f1, e)
            print() 

        print(f"Training completed! Best model F1: {best_f1:.4f} | Best threshold: {best_threshold:.4f}")

        if best_model_state is not None:
            net.load_state_dict(best_model_state)
            net = net.to(device)

    # ---------------- 4. 测试集推理逻辑 ----------------
    print("\n----------------Testing--------------------")
    current_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("./result", exist_ok=True)
    
    if args.epochs == 0 and args.load_model_path:
        save_csv_path = f"./result/submission_test_only_{current_time}.csv"
        best_threshold = load_threshold_metadata(args.load_model_path)
    else:
        save_csv_path = f"./result/submission_{current_time}.csv"
    
    print(f"Running inference on test set with threshold={best_threshold:.4f}...")
    inference_and_save(test_loader, net, device, save_csv_path=save_csv_path, threshold=best_threshold)
    print(f"✅ Inference completed! Results successfully saved to {save_csv_path}\n")