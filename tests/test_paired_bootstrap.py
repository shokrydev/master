import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu

from notebooks.utils.paired_bootstrap import (
    PRIMARY_METRICS,
    PredictionComparison,
    _aggregate_rows_by_patch,
    _bleu4_from_totals,
    _caption_bleu_statistics,
    paired_cluster_bootstrap,
)
from src.evaluation.bentxt_records import BENTxTPrediction


def _row(
    *,
    sample_id: str,
    patch_id: str,
    prediction: str,
    target: str,
    task_type: str = "binary",
    task_category: str = "presence",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "patch_id": patch_id,
        "prediction": prediction,
        "target_texts": [target],
        "task_type": task_type,
        "task_category": task_category,
        "split": "bench",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestPairedBootstrap(unittest.TestCase):
    def test_rows_are_aggregated_by_patch_before_resampling(self) -> None:
        totals, counts = _aggregate_rows_by_patch(
            row_values=np.asarray([1.0, 0.0, 1.0]),
            row_patch_indices=np.asarray([0, 0, 1]),
            n_patches=2,
        )

        np.testing.assert_array_equal(totals, [1.0, 1.0])
        np.testing.assert_array_equal(counts, [2.0, 1.0])

    def test_bleu_statistics_match_evaluation_metric(self) -> None:
        records = [
            BENTxTPrediction(
                prediction="a green field",
                target_texts=("a green field",),
                sample_id="1",
                patch_id="patch-1",
                task_type="captioning",
                task_category="None",
                split="bench",
            ),
            BENTxTPrediction(
                prediction="a forest and river",
                target_texts=("a forest beside a river", "a forest and river"),
                sample_id="2",
                patch_id="patch-2",
                task_type="captioning",
                task_category="None",
                split="bench",
            ),
        ]
        totals = sum(_caption_bleu_statistics(record) for record in records)
        gts = {index: list(record.target_texts) for index, record in enumerate(records)}
        res = {index: [record.prediction] for index, record in enumerate(records)}
        expected = Bleu(4).compute_score(gts, res)[0][3]

        self.assertAlmostEqual(float(_bleu4_from_totals(totals)), expected)

    def test_identical_exports_have_zero_paired_interval(self) -> None:
        rows = [
            _row(sample_id="1", patch_id="patch-a", prediction="yes", target="yes"),
            _row(sample_id="2", patch_id="patch-a", prediction="no", target="yes"),
            _row(sample_id="3", patch_id="patch-b", prediction="yes", target="yes"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.jsonl"
            path_b = Path(tmpdir) / "b.jsonl"
            _write_jsonl(path_a, rows)
            _write_jsonl(path_b, list(reversed(rows)))
            result = paired_cluster_bootstrap(
                PredictionComparison("same", "A", path_a, "B", path_b, "core"),
                metrics=(PRIMARY_METRICS[1],),
                n_resamples=100,
            ).iloc[0]

        self.assertEqual(result["n_rows"], 3)
        self.assertEqual(result["n_patches_total"], 2)
        self.assertEqual(result["difference_a_minus_b"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)

    def test_difference_direction_is_system_a_minus_system_b(self) -> None:
        rows_a = [
            _row(sample_id="1", patch_id="patch-a", prediction="yes", target="yes"),
            _row(sample_id="2", patch_id="patch-a", prediction="yes", target="yes"),
            _row(sample_id="3", patch_id="patch-b", prediction="no", target="yes"),
        ]
        rows_b = [
            _row(sample_id="1", patch_id="patch-a", prediction="no", target="yes"),
            _row(sample_id="2", patch_id="patch-a", prediction="no", target="yes"),
            _row(sample_id="3", patch_id="patch-b", prediction="yes", target="yes"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.jsonl"
            path_b = Path(tmpdir) / "b.jsonl"
            _write_jsonl(path_a, rows_a)
            _write_jsonl(path_b, rows_b)
            result = paired_cluster_bootstrap(
                PredictionComparison("A - B", "A", path_a, "B", path_b, "core"),
                metrics=(PRIMARY_METRICS[1],),
                n_resamples=200,
            ).iloc[0]

        self.assertAlmostEqual(result["score_a"], 2 / 3)
        self.assertAlmostEqual(result["score_b"], 1 / 3)
        self.assertAlmostEqual(result["difference_a_minus_b"], 1 / 3)

    def test_mismatched_pair_metadata_is_rejected(self) -> None:
        row_a = _row(sample_id="1", patch_id="patch-a", prediction="yes", target="yes")
        row_b = _row(sample_id="1", patch_id="patch-b", prediction="yes", target="yes")
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.jsonl"
            path_b = Path(tmpdir) / "b.jsonl"
            _write_jsonl(path_a, [row_a])
            _write_jsonl(path_b, [row_b])

            with self.assertRaisesRegex(ValueError, "Mismatched patch_id"):
                paired_cluster_bootstrap(
                    PredictionComparison("bad", "A", path_a, "B", path_b, "core"),
                    metrics=(PRIMARY_METRICS[1],),
                    n_resamples=10,
                )


if __name__ == "__main__":
    unittest.main()
