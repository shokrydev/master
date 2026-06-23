import unittest
from unittest.mock import patch

from src.metrics.captioning import CaptioningMetrics


class TestCaptioningMetrics(unittest.TestCase):
    def test_remains_hashable_after_accumulating_predictions(self) -> None:
        with patch.object(CaptioningMetrics, "_ensure_meteor_resources"):
            metrics = CaptioningMetrics()

        metrics.update(["prediction"], [["reference"]])

        self.assertIsInstance(hash(metrics), int)


if __name__ == "__main__":
    unittest.main()
