"""Prepare a YOLO dataset from the supplied BSD image and label folders."""

from __future__ import annotations

import random
import os
import shutil
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _normalize_label(label: Path) -> list[str]:
    normalized_lines = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"expected 5 fields in {label} line {line_number}")
        try:
            source_class = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"non-numeric label data in {label} line {line_number}") from error
        if source_class < 0 or any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"invalid YOLO values in {label} line {line_number}")
        fields[0] = "0"
        normalized_lines.append(" ".join(fields))
    return normalized_lines


def prepare_dataset(
    source: Path,
    destination: Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> dict[str, int]:
    """Copy matching image/label pairs into deterministic train/val/test folders."""
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must sum to less than 1")
    image_dir = source / "images"
    label_dir = source / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("source must contain images and labels directories")

    images = {
        image.stem: image
        for image in image_dir.iterdir()
        if image.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {label.stem: label for label in label_dir.glob("*.txt")}
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    pairs = []
    for stem in sorted(set(images) & set(labels)):
        pairs.append((images[stem], labels[stem]))

    if not pairs:
        raise ValueError("no image/label pairs were found")

    random.Random(seed).shuffle(pairs)
    test_count = round(len(pairs) * test_fraction)
    validation_count = round(len(pairs) * validation_fraction)
    splits = {
        "test": pairs[:test_count],
        "val": pairs[test_count : test_count + validation_count],
        "train": pairs[test_count + validation_count :],
    }

    for split, split_pairs in splits.items():
        image_target = destination / "images" / split
        label_target = destination / "labels" / split
        if image_target.exists():
            shutil.rmtree(image_target)
        if label_target.exists():
            shutil.rmtree(label_target)
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            image_output = image_target / image.name
            try:
                os.link(image, image_output)
            except OSError:
                shutil.copy2(image, image_output)
            normalized_lines = _normalize_label(label)
            (label_target / label.name).write_text(
                "\n".join(normalized_lines) + "\n", encoding="utf-8"
            )

    counts = {split: len(split_pairs) for split, split_pairs in splits.items()}
    manifest = {
        "source": str(source.resolve()),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "counts": counts,
        "matched_pairs": len(pairs),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return counts
