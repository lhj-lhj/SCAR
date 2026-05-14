#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_embodiment_classifier import SmallEmbodimentCNN, canonical_embodiment_name


def log_step(message: str) -> None:
    print(f"[embodiment-leakage] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate embodiment leakage on action-transfer videos using a trained "
            "single-frame embodiment classifier."
        )
    )
    parser.add_argument("--classifier-ckpt", required=True, help="Path to best.pt classifier checkpoint.")
    parser.add_argument(
        "--transfer-summary",
        nargs="+",
        required=True,
        help="One or more action-transfer summary.json files.",
    )
    parser.add_argument(
        "--method-label",
        nargs="+",
        default=None,
        help=(
            "Optional display labels, one per --transfer-summary, used when comparing "
            "multiple methods in a single run."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--transferred-pred-only",
        action="store_true",
        help=(
            "If set, evaluate only the predicted future portion of transferred videos. "
            "The start frame defaults to hist_len recorded in each transfer summary."
        ),
    )
    parser.add_argument(
        "--transferred-start-frame",
        type=int,
        default=None,
        help=(
            "Optional explicit start frame for transferred-video evaluation. "
            "Overrides hist_len from the transfer summary when provided."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for aggregated evaluation results. Defaults next to the first summary.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class VideoPrediction:
    num_frames: int
    mean_prob: dict[str, float]
    prob_sum: dict[str, float]
    top1_rate: dict[str, float]


@dataclass(frozen=True)
class FramePreprocess:
    normalize_mean: torch.Tensor | None = None
    normalize_std: torch.Tensor | None = None


def safe_divide(numerator: float, denominator: float, *, eps: float = 1e-8) -> float:
    return float(numerator / max(float(denominator), eps))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload in {path}")
    return payload


def read_video_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def prepare_frames(frames_rgb: np.ndarray, preprocess: FramePreprocess) -> torch.Tensor:
    frames = torch.from_numpy(frames_rgb).to(torch.float32) / 255.0
    frames = frames.permute(0, 3, 1, 2).contiguous()
    if preprocess.normalize_mean is not None and preprocess.normalize_std is not None:
        frames = (frames - preprocess.normalize_mean) / preprocess.normalize_std
    return frames


def load_classifier(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], FramePreprocess]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    class_names = [str(name) for name in checkpoint["class_names"]]
    channels = [int(value) for value in checkpoint["config"]["model"]["channels"]]
    model = SmallEmbodimentCNN(channels=channels, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_cfg = dict(checkpoint.get("config", {}).get("data", {}))
    normalize_mean = data_cfg.get("normalize_mean")
    normalize_std = data_cfg.get("normalize_std")
    preprocess = FramePreprocess()
    if normalize_mean is not None and normalize_std is not None:
        preprocess = FramePreprocess(
            normalize_mean=torch.tensor(normalize_mean, dtype=torch.float32).view(1, 3, 1, 1),
            normalize_std=torch.tensor(normalize_std, dtype=torch.float32).view(1, 3, 1, 1),
        )
    return model, class_names, preprocess


def predict_video(
    model: torch.nn.Module,
    video_path: Path,
    *,
    class_names: list[str],
    preprocess: FramePreprocess,
    device: torch.device,
    batch_size: int,
    start_frame: int = 0,
) -> VideoPrediction:
    frames_rgb = read_video_rgb(video_path)
    start_frame = max(int(start_frame), 0)
    if start_frame >= frames_rgb.shape[0]:
        raise ValueError(
            f"Requested start_frame={start_frame} for video {video_path}, "
            f"but only {frames_rgb.shape[0]} frames are available."
        )
    if start_frame > 0:
        frames_rgb = frames_rgb[start_frame:]
    frames = prepare_frames(frames_rgb, preprocess)
    probs_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            end = min(start + batch_size, frames.shape[0])
            batch = frames[start:end].to(device=device, non_blocking=True)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu()
            probs_chunks.append(probs)
    probs = torch.cat(probs_chunks, dim=0)
    preds = probs.argmax(dim=1)

    mean_prob = probs.mean(dim=0)
    sum_prob = probs.sum(dim=0)
    top1_rate: dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        top1_rate[class_name] = float((preds == class_index).float().mean().item())

    return VideoPrediction(
        num_frames=int(probs.shape[0]),
        mean_prob={class_name: float(mean_prob[idx].item()) for idx, class_name in enumerate(class_names)},
        prob_sum={class_name: float(sum_prob[idx].item()) for idx, class_name in enumerate(class_names)},
        top1_rate=top1_rate,
    )


def aggregate_predictions(
    predictions: list[VideoPrediction],
    *,
    class_names: list[str],
) -> dict[str, Any]:
    total_frames = int(sum(pred.num_frames for pred in predictions))
    if total_frames <= 0:
        raise ValueError("No predictions to aggregate.")

    total_prob_sum = {class_name: 0.0 for class_name in class_names}
    total_top1_count = {class_name: 0.0 for class_name in class_names}
    per_video_mean_prob = {class_name: [] for class_name in class_names}
    per_video_top1_rate = {class_name: [] for class_name in class_names}

    for pred in predictions:
        for class_name in class_names:
            total_prob_sum[class_name] += pred.prob_sum[class_name]
            total_top1_count[class_name] += pred.top1_rate[class_name] * pred.num_frames
            per_video_mean_prob[class_name].append(pred.mean_prob[class_name])
            per_video_top1_rate[class_name].append(pred.top1_rate[class_name])

    return {
        "num_videos": len(predictions),
        "num_frames": total_frames,
        "mean_probability_by_frame": {
            class_name: float(total_prob_sum[class_name] / total_frames)
            for class_name in class_names
        },
        "probability_sum_total": {class_name: float(total_prob_sum[class_name]) for class_name in class_names},
        "top1_rate_by_frame": {
            class_name: float(total_top1_count[class_name] / total_frames)
            for class_name in class_names
        },
        "mean_probability_by_video": {
            class_name: float(np.mean(values)) if values else 0.0
            for class_name, values in per_video_mean_prob.items()
        },
        "mean_top1_rate_by_video": {
            class_name: float(np.mean(values)) if values else 0.0
            for class_name, values in per_video_top1_rate.items()
        },
    }


def build_leakage_metrics(
    *,
    source_class: str,
    target_class: str,
    source_agg: dict[str, Any],
    target_agg: dict[str, Any],
    transferred_agg: dict[str, Any],
) -> dict[str, float | str]:
    source_prob_mean = float(transferred_agg["mean_probability_by_frame"][source_class])
    target_prob_mean = float(transferred_agg["mean_probability_by_frame"][target_class])
    source_prob_sum = float(transferred_agg["probability_sum_total"][source_class])
    target_prob_sum = float(transferred_agg["probability_sum_total"][target_class])
    source_top1_rate = float(transferred_agg["top1_rate_by_frame"][source_class])
    target_top1_rate = float(transferred_agg["top1_rate_by_frame"][target_class])

    source_target_prob_total = source_prob_mean + target_prob_mean
    source_target_top1_total = source_top1_rate + target_top1_rate

    return {
        "source_class": source_class,
        "target_class": target_class,
        "source_video_source_prob_mean": float(
            source_agg["mean_probability_by_frame"][source_class]
        ),
        "source_video_source_top1_rate": float(
            source_agg["top1_rate_by_frame"][source_class]
        ),
        "target_video_target_prob_mean": float(
            target_agg["mean_probability_by_frame"][target_class]
        ),
        "target_video_target_top1_rate": float(
            target_agg["top1_rate_by_frame"][target_class]
        ),
        "transferred_source_prob_mean": source_prob_mean,
        "transferred_target_prob_mean": target_prob_mean,
        "transferred_source_prob_sum": source_prob_sum,
        "transferred_target_prob_sum": target_prob_sum,
        "transferred_source_prob_share_2way": safe_divide(
            source_prob_mean, source_target_prob_total
        ),
        "transferred_target_prob_share_2way": safe_divide(
            target_prob_mean, source_target_prob_total
        ),
        "transferred_target_minus_source_prob": float(target_prob_mean - source_prob_mean),
        "transferred_source_over_target_prob_ratio": safe_divide(
            source_prob_mean, target_prob_mean
        ),
        "transferred_top1_source_rate": source_top1_rate,
        "transferred_top1_target_rate": target_top1_rate,
        "transferred_top1_source_share_2way": safe_divide(
            source_top1_rate, source_target_top1_total
        ),
        "transferred_top1_target_share_2way": safe_divide(
            target_top1_rate, source_target_top1_total
        ),
        "transferred_top1_target_minus_source_rate": float(
            target_top1_rate - source_top1_rate
        ),
    }


def aggregate_result_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"num_results": 0, "by_method": {}}

    metric_keys = [
        "source_video_source_prob_mean",
        "source_video_source_top1_rate",
        "target_video_target_prob_mean",
        "target_video_target_top1_rate",
        "transferred_source_prob_mean",
        "transferred_target_prob_mean",
        "transferred_source_prob_share_2way",
        "transferred_target_prob_share_2way",
        "transferred_target_minus_source_prob",
        "transferred_source_over_target_prob_ratio",
        "transferred_top1_source_rate",
        "transferred_top1_target_rate",
        "transferred_top1_source_share_2way",
        "transferred_top1_target_share_2way",
        "transferred_top1_target_minus_source_rate",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["method_label"]), []).append(result)

    by_method: dict[str, Any] = {}
    for method_label, method_results in grouped.items():
        summary: dict[str, Any] = {
            "num_results": len(method_results),
            "source_to_target_pairs": [
                f"{item['source_class']}->{item['target_class']}" for item in method_results
            ],
        }
        for key in metric_keys:
            values = [float(item["leakage_metrics"][key]) for item in method_results]
            summary[key] = float(np.mean(values))
        by_method[method_label] = summary

    return {
        "num_results": len(results),
        "by_method": by_method,
    }


def evaluate_transfer_summary(
    summary_path: Path,
    *,
    method_label: str,
    model: torch.nn.Module,
    class_names: list[str],
    preprocess: FramePreprocess,
    device: torch.device,
    batch_size: int,
    transferred_pred_only: bool,
    transferred_start_frame: int | None,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    source_class = canonical_embodiment_name(summary["source_dataset_filter"])
    target_class = canonical_embodiment_name(summary["target_dataset_filter"])
    for expected_class in (source_class, target_class):
        if expected_class not in class_names:
            raise KeyError(
                f"Expected class {expected_class!r} from {summary_path}, "
                f"but classifier classes are {class_names}."
            )
    hist_len = int(summary.get("hist_len", 0))
    transferred_eval_start = 0
    if transferred_pred_only:
        transferred_eval_start = (
            int(transferred_start_frame)
            if transferred_start_frame is not None
            else hist_len
        )

    source_predictions: list[VideoPrediction] = []
    target_predictions: list[VideoPrediction] = []
    transferred_predictions: list[VideoPrediction] = []
    pair_details: list[dict[str, Any]] = []

    pairs = list(summary.get("pairs") or [])
    for pair in pairs:
        saved_paths = pair["saved_paths"]
        source_pred = predict_video(
            model,
            Path(saved_paths["source_video"]),
            class_names=class_names,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        )
        target_pred = predict_video(
            model,
            Path(saved_paths["target_video"]),
            class_names=class_names,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        )
        transferred_pred = predict_video(
            model,
            Path(saved_paths["transferred_video"]),
            class_names=class_names,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
            start_frame=transferred_eval_start,
        )
        source_predictions.append(source_pred)
        target_predictions.append(target_pred)
        transferred_predictions.append(transferred_pred)
        pair_details.append(
            {
                "pair_index": int(pair["pair_index"]),
                "source_video": {
                    "num_frames": source_pred.num_frames,
                    "mean_prob": source_pred.mean_prob,
                    "top1_rate": source_pred.top1_rate,
                },
                "target_video": {
                    "num_frames": target_pred.num_frames,
                    "mean_prob": target_pred.mean_prob,
                    "top1_rate": target_pred.top1_rate,
                },
                "transferred_video": {
                    "num_frames": transferred_pred.num_frames,
                    "start_frame": int(transferred_eval_start),
                    "mean_prob": transferred_pred.mean_prob,
                    "top1_rate": transferred_pred.top1_rate,
                },
            }
        )

    source_agg = aggregate_predictions(source_predictions, class_names=class_names)
    target_agg = aggregate_predictions(target_predictions, class_names=class_names)
    transferred_agg = aggregate_predictions(transferred_predictions, class_names=class_names)

    leakage_metrics = build_leakage_metrics(
        source_class=source_class,
        target_class=target_class,
        source_agg=source_agg,
        target_agg=target_agg,
        transferred_agg=transferred_agg,
    )

    return {
        "method_label": method_label,
        "summary_path": str(summary_path),
        "run_dir": summary.get("run_dir"),
        "ckpt_path": summary.get("ckpt_path"),
        "source_split": summary.get("source_split"),
        "target_split": summary.get("target_split"),
        "source_class": source_class,
        "target_class": target_class,
        "num_pairs": len(pairs),
        "hist_len": hist_len,
        "transferred_eval_start_frame": int(transferred_eval_start),
        "class_names": class_names,
        "source_video_aggregate": source_agg,
        "target_video_aggregate": target_agg,
        "transferred_video_aggregate": transferred_agg,
        "leakage_metrics": leakage_metrics,
        "pair_details": pair_details,
    }


def main() -> None:
    args = parse_args()
    classifier_ckpt = Path(args.classifier_ckpt).expanduser().resolve()
    summary_paths = [Path(path).expanduser().resolve() for path in args.transfer_summary]
    if args.method_label is not None and len(args.method_label) != len(summary_paths):
        raise ValueError(
            "--method-label must provide exactly one label per --transfer-summary."
        )
    method_labels = (
        [str(label) for label in args.method_label]
        if args.method_label is not None
        else [path.parent.name for path in summary_paths]
    )
    device = torch.device(args.device)

    model, class_names, preprocess = load_classifier(classifier_ckpt, device)
    log_step(
        f"classifier_ckpt={classifier_ckpt} class_names={class_names} device={device}"
    )
    if preprocess.normalize_mean is not None and preprocess.normalize_std is not None:
        log_step("using checkpoint normalization for video frames")

    results = []
    for method_label, summary_path in zip(method_labels, summary_paths, strict=True):
        log_step(f"evaluating method={method_label} summary={summary_path}")
        result = evaluate_transfer_summary(
            summary_path,
            method_label=method_label,
            model=model,
            class_names=class_names,
            preprocess=preprocess,
            device=device,
            batch_size=int(args.batch_size),
            transferred_pred_only=bool(args.transferred_pred_only),
            transferred_start_frame=args.transferred_start_frame,
        )
        metrics = result["leakage_metrics"]
        log_step(
            f"{method_label} {metrics['source_class']}->{metrics['target_class']}: "
            f"mean_p(source)={metrics['transferred_source_prob_mean']:.4f} "
            f"mean_p(target)={metrics['transferred_target_prob_mean']:.4f} "
            f"share2(target)={metrics['transferred_target_prob_share_2way']:.4f} "
            f"delta(target-source)={metrics['transferred_target_minus_source_prob']:.4f} "
            f"top1_source={metrics['transferred_top1_source_rate']:.4f} "
            f"top1_target={metrics['transferred_top1_target_rate']:.4f}"
        )
        results.append(result)

    aggregate = aggregate_result_metrics(results)
    for method_label, method_summary in aggregate["by_method"].items():
        log_step(
            f"aggregate {method_label}: "
            f"target_share2={method_summary['transferred_target_prob_share_2way']:.4f} "
            f"source_share2={method_summary['transferred_source_prob_share_2way']:.4f} "
            f"target_minus_source={method_summary['transferred_target_minus_source_prob']:.4f} "
            f"top1_target_minus_source={method_summary['transferred_top1_target_minus_source_rate']:.4f}"
        )

    payload = {
        "classifier_ckpt": str(classifier_ckpt),
        "class_names": class_names,
        "transferred_pred_only": bool(args.transferred_pred_only),
        "transferred_start_frame": args.transferred_start_frame,
        "results": results,
        "aggregate": aggregate,
    }

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
    else:
        output_json = summary_paths[0].parent / "embodiment_leakage_eval.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_step(f"saved={output_json}")


if __name__ == "__main__":
    main()
