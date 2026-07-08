import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bentxt_records import BENTxTPrediction, TaskType, load_predictions_jsonl
from src.evaluation.bentxt_scoring import (
    evaluate_predictions,
    score_prediction,
    score_predictions,
    summarize_scores,
)


def _prediction(
    *,
    prediction: str,
    target: str,
    task_type: TaskType,
    task_category: str = "category",
    sample_id: str = "sample",
) -> BENTxTPrediction:
    return BENTxTPrediction(
        prediction=prediction,
        target_texts=(target,),
        sample_id=sample_id,
        patch_id=f"patch-{sample_id}",
        task_type=task_type,
        task_category=task_category,
        split="bench",
        country="Germany",
        season="summer",
        climate_zone="temperate",
        location_condition="loc_embed",
        model_size="2B",
    )


class TestBENTxTScoring(unittest.TestCase):
    def test_load_predictions_jsonl_validates_required_fields(self) -> None:
        row = {
            "prediction": "yes",
            "target_texts": ["yes"],
            "sample_id": "row-1",
            "patch_id": "patch-1",
            "task_type": "binary",
            "task_category": "presence",
            "split": "bench",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = load_predictions_jsonl(path)

        self.assertEqual(records[0].sample_id, "row-1")
        self.assertEqual(records[0].target_texts, ("yes",))

    def test_binary_and_mcq_accuracy_treat_unparsed_answers_as_wrong(self) -> None:
        records = [
            _prediction(prediction="yes", target="yes", task_type="binary", sample_id="1"),
            _prediction(prediction="yes.", target="yes", task_type="binary", sample_id="2"),
            _prediction(prediction="b", target="a", task_type="mcq", sample_id="3"),
            _prediction(prediction="a", target="a", task_type="mcq", sample_id="4"),
        ]
        rows = summarize_scores(score_predictions(records), group_by=("task_type",))
        by_type = {row["task_type"]: row for row in rows}

        self.assertEqual(by_type["binary"]["n"], 2)
        self.assertEqual(by_type["binary"]["correct"], 1)
        self.assertAlmostEqual(by_type["binary"]["accuracy"], 0.5)
        self.assertAlmostEqual(by_type["binary"]["extraction_success"], 0.5)

        self.assertEqual(by_type["mcq"]["correct"], 1)
        self.assertAlmostEqual(by_type["mcq"]["accuracy"], 0.5)
        self.assertAlmostEqual(by_type["mcq"]["extraction_success"], 1.0)

    def test_bbox_summary_reports_iou_and_threshold_accuracy(self) -> None:
        records = [
            _prediction(
                prediction="[0.0 0.0, 1.0 1.0]",
                target="[0.0 0.0, 1.0 1.0]",
                task_type="bounding box",
                sample_id="1",
            ),
            _prediction(
                prediction="[0.0, 0.0, 1.0, 1.0]",
                target="[0.0 0.0, 1.0 1.0]",
                task_type="bounding box",
                sample_id="2",
            ),
        ]
        rows = summarize_scores(score_predictions(records), group_by=("task_type",))
        bbox = rows[0]

        self.assertEqual(bbox["task_type"], "bounding box")
        self.assertAlmostEqual(bbox["extraction_success"], 0.5)
        self.assertAlmostEqual(bbox["miou"], 0.5)
        self.assertAlmostEqual(bbox["acc@50"], 0.5)
        self.assertAlmostEqual(bbox["acc@90"], 0.5)

    def test_evaluate_predictions_includes_standard_stratifications(self) -> None:
        records = [
            _prediction(prediction="yes", target="yes", task_type="binary", sample_id="1"),
            _prediction(prediction="a", target="a", task_type="mcq", sample_id="2"),
        ]
        summary = evaluate_predictions(records)

        self.assertIn("captioning", summary)
        self.assertIn("by_task_type", summary)
        self.assertIn("by_task_category", summary)
        self.assertIn("by_country", summary)

    def test_summary_requires_task_type_grouping(self) -> None:
        scores = score_predictions(
            [_prediction(prediction="yes", target="yes", task_type="binary")]
        )

        with self.assertRaisesRegex(ValueError, "task_type"):
            summarize_scores(scores, group_by=("country",))

    def test_score_prediction_keeps_caption_rows_parse_neutral(self) -> None:
        score = score_prediction(
            _prediction(
                prediction="A caption.",
                target="A reference.",
                task_type="captioning",
                sample_id="caption",
            )
        )

        self.assertTrue(score.extracted)
        self.assertIsNone(score.correct)


if __name__ == "__main__":
    unittest.main()
