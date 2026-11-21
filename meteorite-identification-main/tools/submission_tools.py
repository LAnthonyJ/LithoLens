from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def predict_on_test(model, loader, device):
    """
    Run inference on test images and return a {id: pred_label} dictionary.

    Why return a dictionary before writing csv:
    - Prediction and submission formatting are separate steps
    - A dictionary simplifies inspection
    - Scores can then be thresholded, blended or averaged over TTA views
    """
    model.eval()
    id_to_pred = {}

    for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
        # Move the test batch to the device
        images = images.to(device, non_blocking=True)

        # logits typically has shape [batch_size, num_classes]
        logits = model(images)

        # Use the class with the highest score
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for pred, path in zip(preds, img_paths):
            # Kaggle Use the filename as the submission ID
            image_id = Path(path).name
            id_to_pred[image_id] = int(pred)

    return id_to_pred


def make_submission(template_csv_path, id_to_pred, output_path):
    """
    Build the submission using sample_submission.csv.

    Using the template ensures:
    - consistent column names
    - the original test order
    - detection of missing predictions
    """
    template_csv_path = Path(template_csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(template_csv_path)
    if "id" not in df.columns:
        raise ValueError("submission Template is missing the id column")

    # Look up predictions by image ID
    df["label"] = df["id"].map(id_to_pred)

    # Null entries indicate missing predictions
    if df["label"].isna().any():
        missing_ids = df.loc[df["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for test images: {missing_ids}")

    # Kaggle Cast labels to integers for submission
    df["label"] = df["label"].astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
