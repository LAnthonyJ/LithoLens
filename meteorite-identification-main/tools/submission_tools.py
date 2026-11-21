from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def predict_on_test(model, loader, device):
    """
    在测试集上跑推理，返回 {id: pred_label} 字典。

    为什么这里返回字典，而不是直接写 csv：
    - 推理和生成提交文件本质上是两个步骤
    - 字典格式更方便调试
    - 如果后面要做阈值调整、多模型融合、TTA，也更容易扩展
    """
    model.eval()
    id_to_pred = {}

    for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
        # 把一批测试图片送到设备上
        images = images.to(device, non_blocking=True)

        # logits 形状通常是 [batch_size, num_classes]
        logits = model(images)

        # 选择分数最大的类别作为预测类别
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for pred, path in zip(preds, img_paths):
            # Kaggle 提交只需要图片文件名，不需要完整路径
            image_id = Path(path).name
            id_to_pred[image_id] = int(pred)

    return id_to_pred


def make_submission(template_csv_path, id_to_pred, output_path):
    """
    依据 sample_submission.csv 模板生成提交文件。

    使用模板文件的原因是：
    - 可以保证列名正确
    - 可以保证测试图片顺序与官方要求一致
    - 可以检查是否有测试样本被漏掉
    """
    template_csv_path = Path(template_csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(template_csv_path)
    if "id" not in df.columns:
        raise ValueError("submission 模板缺少 id 列")

    # 使用图片 id 去字典里查找预测结果
    df["label"] = df["id"].map(id_to_pred)

    # 如果存在空值，说明有测试图片没有得到预测结果
    if df["label"].isna().any():
        missing_ids = df.loc[df["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"以下测试图片没有预测结果: {missing_ids}")

    # Kaggle 提交通常要求 label 为整数
    df["label"] = df["label"].astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
