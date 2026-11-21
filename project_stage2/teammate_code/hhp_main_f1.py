import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from data_loader.utils import build_all_datasets
from models.utils import build_model
def make_exp_name(args):
    if args.exp_name:
        return args.exp_name
    return (
        f"{args.model}_f1"
        f"_lr{args.learning_rate}"
        f"_wd{args.decay}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
    )
def threshold_search(y_true, prob, start=0.05, end=0.95, step=0.005):
    thresholds = np.arange(start, end + 1e-9, step)
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for th in thresholds:
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(th),
                "f1": float(f1),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }

    return best
def evaluate(model, loader, criterion, device, use_amp=True, desc="Val"):
    model.eval()

    total_loss = 0.0
    total_n = 0
    all_prob = []
    all_label = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            prob = torch.softmax(logits.float(), dim=1)[:, 1]

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            all_prob.append(prob.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())

    prob = np.concatenate(all_prob)
    y_true = np.concatenate(all_label)

    pred_05 = (prob >= 0.5).astype(int)

    metrics_05 = {
        "loss": float(total_loss / max(total_n, 1)),
        "f1_05": float(f1_score(y_true, pred_05, zero_division=0)),
        "acc_05": float(accuracy_score(y_true, pred_05)),
        "precision_05": float(precision_score(y_true, pred_05, zero_division=0)),
        "recall_05": float(recall_score(y_true, pred_05, zero_division=0)),
    }

    best_th = threshold_search(y_true, prob)
    metrics = {**metrics_05, **{f"best_{k}": v for k, v in best_th.items()}}

    return metrics, y_true, prob
def predict_test(model, loader, device, use_amp=True, desc="Test"):
    model.eval()

    all_id = []
    all_prob = []

    with torch.no_grad():
        for images, img_names in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(images)

            prob = torch.softmax(logits.float(), dim=1)[:, 1]

            all_id.extend(list(img_names))
            all_prob.append(prob.detach().cpu().numpy())

    prob = np.concatenate(all_prob)
    return all_id, prob
def save_head_checkpoint(model, path, args, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": args.model,
        "epoch": epoch,
        "head_state_dict": model.head.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(ckpt, path)
def load_head_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if "head_state_dict" in ckpt:
        model.head.load_state_dict(ckpt["head_state_dict"])
    else:
        raise KeyError("Checkpoint does not contain 'head_state_dict'.")

    return ckpt
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--data_path", type=str, default="/data/final project/data/raw")
    parser.add_argument("--output_root", type=str, default="/data/final project/experiments")

    parser.add_argument("--model", type=str, default="dinov3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--decay", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--load_head_path", type=str, default=None)

    args = parser.parse_args()

    use_amp = not args.no_amp
    exp_name = make_exp_name(args)
    exp_dir = Path(args.output_root) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment:", exp_name)
    print("Output dir:", exp_dir)
    print("Args:", vars(args))

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True

    train_data, val_data, test_data, num_classes = build_all_datasets(
        args.data_path,
        args,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    model = build_model(args, device=device, num_classes=num_classes)

    if args.load_head_path:
        ckpt = load_head_checkpoint(model, args.load_head_path, device)
        print("Loaded head checkpoint:", args.load_head_path)
        print("Checkpoint metrics:", ckpt.get("metrics", {}))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable_params))

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    best_val_true = None
    best_val_prob = None
    no_improve = 0

    metrics_rows = []

    best_head_path = exp_dir / "best_head.pth"
    last_head_path = exp_dir / "last_head.pth"

    if args.epochs > 0:
        for epoch in range(1, args.epochs + 1):
            model.train()

            total_loss = 0.0
            total_n = 0
            total_correct = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_n += bs

                pred = torch.argmax(logits.detach(), dim=1)
                total_correct += (pred == labels).sum().item()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()

            train_loss = total_loss / max(total_n, 1)
            train_acc = total_correct / max(total_n, 1)

            val_metrics, val_true, val_prob = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
                desc=f"Epoch {epoch}/{args.epochs} [Val]"
            )

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }
            metrics_rows.append(row)
            pd.DataFrame(metrics_rows).to_csv(exp_dir / "metrics.csv", index=False)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1@0.5={val_metrics['f1_05']:.4f} | "
                f"val_best_f1={val_metrics['best_f1']:.4f} | "
                f"best_th={val_metrics['best_threshold']:.3f} | "
                f"precision={val_metrics['best_precision']:.4f} | "
                f"recall={val_metrics['best_recall']:.4f}"
            )

            save_head_checkpoint(model, last_head_path, args, epoch, val_metrics)

            if val_metrics["best_f1"] > best_f1:
                best_f1 = val_metrics["best_f1"]
                best_epoch = epoch
                best_metrics = val_metrics
                best_val_true = val_true
                best_val_prob = val_prob
                no_improve = 0

                save_head_checkpoint(model, best_head_path, args, epoch, val_metrics)

                print(
                    f"New best by F1 | epoch={epoch} | "
                    f"best_f1={best_f1:.4f} | "
                    f"threshold={val_metrics['best_threshold']:.3f}"
                )
            else:
                no_improve += 1
                print(f"No F1 improvement: {no_improve}/{args.patience}")

                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
                    break

            print()

    if best_head_path.exists():
        ckpt = load_head_checkpoint(model, best_head_path, device)
        best_metrics = ckpt["metrics"]
        print("Loaded best head for final inference:", best_head_path)
    else:
        val_metrics, best_val_true, best_val_prob = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp, desc="Final Val"
        )
        best_metrics = val_metrics

    best_threshold = float(best_metrics["best_threshold"])
    print("Final threshold from validation:", best_threshold)

    if best_val_true is not None and best_val_prob is not None:
        val_df = pd.DataFrame({
            "label": best_val_true.astype(int),
            "prob": best_val_prob.astype(float),
        })
        val_df["pred"] = (val_df["prob"] >= best_threshold).astype(int)
        val_df.to_csv(exp_dir / "val_prob_best.csv", index=False)

    test_ids, test_prob = predict_test(
        model,
        test_loader,
        device,
        use_amp=use_amp,
        desc="Test"
    )

    prob_df = pd.DataFrame({
        "id": test_ids,
        "prob": test_prob.astype(float),
    })
    prob_df.to_csv(exp_dir / "test_prob.csv", index=False)

    sub_df = pd.DataFrame({
        "id": test_ids,
        "label": (test_prob >= best_threshold).astype(int),
    })
    sub_df.to_csv(exp_dir / "submission.csv", index=False)

    summary = {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_threshold": best_threshold,
        "submission_path": str(exp_dir / "submission.csv"),
        "test_prob_path": str(exp_dir / "test_prob.csv"),
        "best_head_path": str(best_head_path),
    }

    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()