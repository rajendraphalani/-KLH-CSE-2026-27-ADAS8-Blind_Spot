"""CLI for preparing data, training, and running blind-spot detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import prepare_dataset


PROJECT_ROOT = Path(__file__).parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADAS blind-spot object detection")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="split a YOLO dataset into train/val/test")
    prepare.add_argument("--source", type=Path, required=True, help="BSD Dataset directory")
    prepare.add_argument("--destination", type=Path, default=PROJECT_ROOT / "dataset" / "processed")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=fraction, default=0.2)
    prepare.add_argument("--test-fraction", type=fraction, default=0.1)

    train = commands.add_parser("train", help="train the detector")
    train.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset.yaml")
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--epochs", type=positive_int, default=150)
    train.add_argument("--imgsz", type=positive_int, default=960)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--batch", type=positive_int, default=16)
    train.add_argument("--fraction", type=fraction, default=1.0)
    train.add_argument("--name", default="bsd-baseline")
    train.add_argument("--device", help="training device, for example 0 for the first Colab GPU")
    train.add_argument("--cache", action="store_true", help="cache images in RAM for faster training")
    train.add_argument("--patience", type=int, default=35, help="epochs without improvement before stopping")
    train.add_argument("--optimizer", choices=("auto", "SGD", "Adam", "AdamW"), default="auto")
    train.add_argument("--cos-lr", action="store_true", help="use cosine learning-rate decay")
    train.add_argument("--close-mosaic", type=int, default=15, help="disable mosaic augmentation for the final epochs")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lr0", type=float, default=0.01)
    train.add_argument("--lrf", type=float, default=0.01)
    train.add_argument("--weight-decay", type=float, default=0.0005)
    train.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")

    evaluate = commands.add_parser("evaluate", help="evaluate a detector on a dataset split")
    evaluate.add_argument("--data", type=Path, default=PROJECT_ROOT / "dataset.yaml")
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--imgsz", type=positive_int, default=960)
    evaluate.add_argument("--batch", type=positive_int, default=16)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", help="evaluation device, for example 0 for the first GPU")
    evaluate.add_argument("--output", type=Path, help="optional JSON metrics output path")

    predict = commands.add_parser("predict", help="run inference on an image or directory")
    predict.add_argument("source", type=Path)
    predict.add_argument("--weights", type=Path, required=True)
    predict.add_argument("--conf", type=fraction, default=0.35)
    predict.add_argument("--imgsz", type=positive_int, default=960)
    predict.add_argument("--device", help="inference device, for example 0 for the first GPU")
    predict.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        counts = prepare_dataset(
            args.source,
            args.destination,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
        print("Prepared dataset: " + ", ".join(f"{split}={count}" for split, count in counts.items()))
        return

    from ultralytics import YOLO

    if args.command == "train":
        model = YOLO(args.model)
        train_options = {
            "data": str(args.data),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "batch": args.batch,
            "fraction": args.fraction,
            "name": args.name,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "cos_lr": args.cos_lr,
            "close_mosaic": args.close_mosaic,
            "seed": args.seed,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "weight_decay": args.weight_decay,
            "cache": args.cache,
            "project": str(args.project),
        }
        if args.device is not None:
            train_options["device"] = args.device
        model.train(**train_options)
    elif args.command == "evaluate":
        model = YOLO(str(args.weights))
        evaluation_options = {
            "data": str(args.data),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "plots": False,
        }
        if args.device is not None:
            evaluation_options["device"] = args.device
        metrics = model.val(**evaluation_options)
        report = {
            "data": str(args.data),
            "weights": str(args.weights),
            "split": args.split,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
        print("Evaluation: " + ", ".join(f"{key}={value:.4f}" for key, value in report.items() if isinstance(value, float)))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        model = YOLO(str(args.weights))
        prediction_options = {
            "source": str(args.source),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "project": str(args.project),
            "save": True,
        }
        if args.device is not None:
            prediction_options["device"] = args.device
        model.predict(**prediction_options)


if __name__ == "__main__":
    main()
