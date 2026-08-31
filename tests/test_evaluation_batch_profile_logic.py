import unittest

from scripts.evaluation_batch_profile_logic import (
    evenly_spaced_indices,
    recommend_throughput_batch,
    recommend_worker_count,
    refinement_batch,
    safe_capacity_batches,
)


class EvaluationBatchProfileLogicTest(unittest.TestCase):
    def test_evenly_spaced_indices_cover_the_population(self):
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 2, 5, 7])
        self.assertEqual(evenly_spaced_indices(3, 5), [0, 1, 2])

    def test_refinement_batch_binary_searches_to_resolution(self):
        self.assertEqual(refinement_batch(256, 512, 32), 384)
        self.assertEqual(refinement_batch(384, 512, 32), 448)
        self.assertIsNone(refinement_batch(480, 512, 32))

    def test_capacity_requires_success_and_memory_margin(self):
        results = [
            {"batch_size": 32, "status": "ok", "peak_reserved_gb": 40.0},
            {"batch_size": 64, "status": "ok", "peak_reserved_gb": 67.0},
            {"batch_size": 128, "status": "ok", "peak_reserved_gb": 69.0},
            {"batch_size": 256, "status": "oom"},
        ]
        self.assertEqual(
            safe_capacity_batches(
                results,
                total_memory_gb=80.0,
                safety_fraction=0.85,
            ),
            [32, 64],
        )

    def test_recommendation_uses_smallest_near_best_safe_batch(self):
        results = [
            {"batch_size": 16, "status": "ok", "samples_per_second": 4.0},
            {"batch_size": 32, "status": "ok", "samples_per_second": 7.9},
            {"batch_size": 64, "status": "ok", "samples_per_second": 8.0},
            {"batch_size": 128, "status": "ok", "samples_per_second": 9.0},
        ]
        self.assertEqual(
            recommend_throughput_batch(
                results,
                safe_batches=[16, 32, 64],
                near_best_fraction=0.98,
            ),
            32,
        )

    def test_recommendation_is_none_without_safe_success(self):
        self.assertIsNone(
            recommend_throughput_batch(
                [{"batch_size": 16, "status": "oom"}],
                safe_batches=[],
                near_best_fraction=0.98,
            )
        )

    def test_worker_recommendation_uses_smallest_near_best_count(self):
        results = [
            {"num_workers": 8, "status": "ok", "samples_per_second": 9.9},
            {"num_workers": 9, "status": "oom"},
            {"num_workers": 10, "status": "ok", "samples_per_second": 10.0},
            {"num_workers": 12, "status": "ok", "samples_per_second": 9.8},
        ]
        self.assertEqual(
            recommend_worker_count(results, near_best_fraction=0.98),
            8,
        )


if __name__ == "__main__":
    unittest.main()
