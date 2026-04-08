import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data_modules.gaia_datamodule import GAIADataModule


def _make_png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_gaia_split(
    root: Path,
    split_dir: str,
    manifest_name: str,
    sample_key: str,
    image_id: str,
    captions: list[str],
    lat: float,
    lon: float,
) -> None:
    shard_dir = root / split_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        f"{sample_key}.json": json.dumps({"id": image_id, "captions": captions}).encode("utf-8"),
        f"{sample_key}.txt": captions[0].encode("utf-8"),
        f"{sample_key}.png": _make_png_bytes((12, 34, 56)),
    }

    with tarfile.open(shard_dir / "00000.tar", "w") as tar:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    manifest = [{"id": image_id, "captions": captions, "lat": lat, "lon": lon}]
    (root / manifest_name).write_text(json.dumps(manifest))


class TestGAIADataModule(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

        _write_gaia_split(
            root=self.root,
            split_dir="train",
            manifest_name="train_data.json",
            sample_key="000000",
            image_id="train-sample",
            captions=["train a", "train b"],
            lat=10.5,
            lon=20.5,
        )
        _write_gaia_split(
            root=self.root,
            split_dir="val",
            manifest_name="val_data.json",
            sample_key="000001",
            image_id="val-sample",
            captions=["val a", "val b"],
            lat=-1.0,
            lon=42.0,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_datamodule(self) -> GAIADataModule:
        return GAIADataModule(
            gaia_root=str(self.root),
            batch_size=1,
            num_workers=0,
            multi_caption=True,
            lat_column="lat",
            lon_column="lon",
            system_prompt="system",
            user_prompt="Describe at {lat:.1f}, {lon:.1f}.",
            pin_memory=False,
            persistent_workers=False,
        )

    def test_fit_setup_reads_official_gaia_root_layout(self) -> None:
        dm = self._make_datamodule()
        dm.setup("fit")

        train_item = next(iter(dm.train_dataset))
        val_item = next(iter(dm.val_dataset))

        self.assertEqual(train_item["image_id"], "train-sample")
        self.assertEqual(train_item["references"], ["train a", "train b"])
        self.assertEqual(train_item["lat"], 10.5)
        self.assertEqual(train_item["lon"], 20.5)
        self.assertIn("Describe at 10.5, 20.5.", train_item["messages"][1]["content"][0]["text"])

        self.assertEqual(val_item["image_id"], "val-sample")
        self.assertEqual(val_item["references"], ["val a", "val b"])

    def test_train_dataloader_preserves_model_facing_batch_shape(self) -> None:
        dm = self._make_datamodule()
        dm.set_collator(lambda batch: batch)
        dm.setup("fit")

        batch = list(dm.train_dataloader())[0]

        self.assertEqual(len(batch), 1)
        self.assertIn("messages", batch[0])
        self.assertIn("references", batch[0])
        self.assertIn("image_id", batch[0])

    def test_test_stage_requires_local_test_split_files(self) -> None:
        dm = self._make_datamodule()

        with self.assertRaises(FileNotFoundError):
            dm.setup("test")


if __name__ == "__main__":
    unittest.main()
