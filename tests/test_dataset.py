import json
import unittest
from pathlib import Path

from main import build_parser
from src.dataset import prepare_dataset


def make_source(root: Path, *, include_orphan: bool = False) -> Path:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(10):
        (image_dir / f"frame({index}).jpg").write_bytes(b"image")
        (label_dir / f"frame({index}).txt").write_text(
            "2 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    if include_orphan:
        (label_dir / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return root


class DatasetTests(unittest.TestCase):
    def test_prepare_is_deterministic_and_reports_orphans(self) -> None:
        with self.subTest("temporary dataset"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                source = make_source(tmp_path / "source", include_orphan=True)
                first = tmp_path / "first"
                second = tmp_path / "second"

                self.assertEqual(
                    prepare_dataset(source, first, seed=7),
                    prepare_dataset(source, second, seed=7),
                )
                first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
                second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(first_manifest["orphan_labels"], ["orphan"])
                self.assertEqual(first_manifest["counts"], second_manifest["counts"])
                self.assertEqual(
                    sorted(path.name for path in (first / "images" / "train").iterdir()),
                    sorted(path.name for path in (second / "images" / "train").iterdir()),
                )
                normalized = next((first / "labels").glob("*/*.txt")).read_text(encoding="utf-8")
                self.assertTrue(normalized.startswith("0 "))


    def test_prepare_rejects_invalid_label(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = make_source(tmp_path / "source")
            (source / "labels" / "frame(0).txt").write_text(
                "0 1.2 0.5 0.2 0.2\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "invalid YOLO values"):
                prepare_dataset(source, tmp_path / "output")


    def test_cli_exposes_evaluation_output(self) -> None:
        args = build_parser().parse_args(
            ["evaluate", "--weights", "best.pt", "--output", "metrics.json"]
        )

        self.assertEqual(args.split, "test")
        self.assertEqual(args.output, Path("metrics.json"))


if __name__ == "__main__":
    unittest.main()