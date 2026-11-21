import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_utils import make_submission_topk_df, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Blend stage2 prediction outputs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="Output directories to blend, e.g. outputs/stage2 outputs/stage2_effv2",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "stage2_blend",
    )
    parser.add_argument(
        "--method",
        choices=["rank", "prob", "logit"],
        default="rank",
        help="How to combine predictions. rank is safest when model calibration differs.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated per-run weights. Default: equal weights.",
    )
    parser.add_argument(
        "--top-counts",
        type=str,
        default="91,92,93",
        help="Comma-separated positive counts for candidate submission files.",
    )
    parser.add_argument(
        "--primary-top",
        type=int,
        default=92,
        help="Positive count used for output-dir/submission.csv.",
    )
    return parser.parse_args()


def parse_counts(text, n_items):
    counts = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        count = max(0, min(count, n_items))
        if count not in counts:
            counts.append(count)
    return counts


def parse_weights(text, n_runs):
    if not str(text).strip():
        return [1.0] * n_runs
    weights = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(weights) != n_runs:
        raise ValueError(
            f"--weights has {len(weights)} values, but --runs has {n_runs} directories"
        )
    if any(w < 0 for w in weights):
        raise ValueError("--weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("At least one --weights value must be positive")
    return weights


def transform_scores(probs, method):
    probs = pd.Series(probs, dtype="float64")
    if method == "rank":
        return probs.rank(method="average", ascending=True, pct=True).to_numpy()
    if method == "prob":
        return probs.to_numpy()
    clipped = np.clip(probs.to_numpy(), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def load_template_ids(data_root):
    template_path = Path(data_root) / "sample_submission.csv"
    df = pd.read_csv(template_path)
    if "id" not in df.columns:
        raise ValueError(f"{template_path} must contain an id column")
    return template_path, df["id"].astype(str).tolist()


def load_run_score(run_dir, method, template_ids, run_index):
    run_dir = Path(run_dir)
    components_path = run_dir / "test_components.csv"
    fallback_path = run_dir / "test_predictions.csv"
    run_name = run_dir.name or f"run{run_index}"

    if components_path.exists():
        df = pd.read_csv(components_path)
        required = {"id", "prob", "component"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{components_path} missing columns: {sorted(missing)}")

        component_scores = []
        component_names = []
        for component, part in df.groupby("component", sort=False):
            part = part.copy()
            part["id"] = part["id"].astype(str)
            series = part.set_index("id")["prob"].astype(float).reindex(template_ids)
            if series.isna().any():
                missing_ids = series[series.isna()].index[:5].tolist()
                raise RuntimeError(f"{components_path} missing ids: {missing_ids}")
            component_scores.append(transform_scores(series.to_numpy(), method))
            component_names.append(str(component))

        matrix = np.vstack(component_scores)
        score = matrix.mean(axis=0)
        source_path = components_path
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
        if "id" not in df.columns or "prob" not in df.columns:
            raise ValueError(f"{fallback_path} must contain id and prob columns")
        series = df.assign(id=df["id"].astype(str)).set_index("id")["prob"].astype(float)
        series = series.reindex(template_ids)
        if series.isna().any():
            missing_ids = series[series.isna()].index[:5].tolist()
            raise RuntimeError(f"{fallback_path} missing ids: {missing_ids}")
        score = transform_scores(series.to_numpy(), method)
        component_names = [f"{run_name}:test_predictions"]
        source_path = fallback_path
    else:
        raise FileNotFoundError(
            f"{run_dir} has neither test_components.csv nor test_predictions.csv"
        )

    return {
        "run": str(run_dir),
        "run_name": run_name,
        "score": score,
        "source_path": str(source_path),
        "component_count": len(component_names),
        "components": component_names,
    }


def threshold_for_topk(scores, positive_count):
    scores = np.asarray(scores, dtype=np.float64)
    k = max(0, min(int(positive_count), len(scores)))
    if k <= 0:
        return float("inf")
    return float(np.sort(scores)[-k])


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_path, template_ids = load_template_ids(args.data_root)
    weights = parse_weights(args.weights, len(args.runs))
    run_payloads = [
        load_run_score(run_dir, args.method, template_ids, idx)
        for idx, run_dir in enumerate(args.runs, start=1)
    ]

    weight_sum = float(sum(weights))
    blend_score = np.zeros(len(template_ids), dtype=np.float64)
    for weight, payload in zip(weights, run_payloads):
        blend_score += float(weight) * np.asarray(payload["score"], dtype=np.float64)
    blend_score /= weight_sum

    pred_df = pd.DataFrame({"id": template_ids, "score": blend_score})
    pred_df.to_csv(args.output_dir / "blended_predictions.csv", index=False)

    ids_to_score = dict(zip(template_ids, blend_score.tolist()))
    primary_df = make_submission_topk_df(
        template_csv_path=template_path,
        ids_to_prob=ids_to_score,
        positive_count=args.primary_top,
    )
    primary_path = args.output_dir / "submission.csv"
    primary_df.to_csv(primary_path, index=False)

    candidate_rows = []
    counts = parse_counts(args.top_counts, len(template_ids))
    if args.primary_top not in counts:
        counts.append(max(0, min(int(args.primary_top), len(template_ids))))
    for count in counts:
        submission_df = make_submission_topk_df(
            template_csv_path=template_path,
            ids_to_prob=ids_to_score,
            positive_count=count,
        )
        path = args.output_dir / f"submission_top{count}.csv"
        submission_df.to_csv(path, index=False)
        candidate_rows.append(
            {
                "positive_count": int(count),
                "positive_rate": float(count / max(1, len(submission_df))),
                "equivalent_score": threshold_for_topk(blend_score, count),
                "path": str(path),
            }
        )

    metrics = {
        "method": args.method,
        "runs": [
            {
                "run": payload["run"],
                "source_path": payload["source_path"],
                "weight": float(weight),
                "component_count": int(payload["component_count"]),
            }
            for weight, payload in zip(weights, run_payloads)
        ],
        "primary_top": int(args.primary_top),
        "primary_positive_count": int(primary_df["label"].sum()),
        "candidate_submissions": candidate_rows,
        "prediction_path": str(args.output_dir / "blended_predictions.csv"),
        "submission_path": str(primary_path),
    }
    save_json(args.output_dir / "blend_metrics.json", metrics)

    print(f"Blended predictions: {args.output_dir / 'blended_predictions.csv'}")
    print(f"Submission: {primary_path}")
    print(
        f"Primary positives: {int(primary_df['label'].sum())}/{len(primary_df)} "
        f"(top{args.primary_top})"
    )
    if candidate_rows:
        print(
            "Candidate top-k files: "
            + ", ".join(str(row["positive_count"]) for row in candidate_rows)
        )


if __name__ == "__main__":
    main()
