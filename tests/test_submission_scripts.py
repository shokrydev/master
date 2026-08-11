import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class SubmissionScriptsTest(unittest.TestCase):
    def test_early_pipeline_submits_correct_and_shuffled_evaluations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / ".env").write_text(
                "\n".join(
                    (
                        "BIGEARTHNET_V2_LMDB_ROOT=/tmp/bentxt.lmdb",
                        "BIGEARTHNET_TXT_PARQUET_PATH=/tmp/bentxt.parquet",
                        "BIGEARTHNET_ENCODER_DIR=/tmp/encoder",
                        "FINETUNING_OUTPUT_ROOT=/tmp/finetuning",
                        "EVALUATION_OUTPUT_ROOT=/tmp/evaluation",
                        "HF_HOME=/tmp/hf",
                        "SATCLIP_CHECKPOINT_PATH=/tmp/l10.ckpt",
                        "SATCLIP_L40_CHECKPOINT_PATH=/tmp/l40.ckpt",
                        "SLURM_DEFAULT_PARTITION=big_job",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                "if [ \"${1:-}\" = \"--parsable\" ]; then\n"
                "    echo 98765\n"
                "else\n"
                "    echo 'Submitted batch job 98766'\n"
                "fi\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    str(REPO_ROOT / "scripts/submit_early_convergence_job.sh"),
                    "--condition",
                    "loc_embed",
                    "--name",
                    "loc-embed-marker-2B-1000",
                    "--submit-evaluations",
                    "--config",
                    "configs/finetuning/ablations/loc_embed_satclip_l40.yaml",
                    "--config",
                    "configs/finetuning/ablations/loc_embed_geolocation_marker.yaml",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("Submitted batch job"), 3)
            self.assertEqual(
                result.stdout.count("--dependency=afterok:98765"),
                2,
            )
            self.assertIn(
                "/tmp/finetuning/bigearthnet_98765/qlora_adapter",
                result.stdout,
            )
            self.assertIn(
                "--data.init_args.coordinate_perturbation shuffled",
                result.stdout,
            )
            self.assertGreaterEqual(
                result.stdout.count("loc_embed_satclip_l40.yaml"),
                3,
            )
            self.assertGreaterEqual(
                result.stdout.count("loc_embed_geolocation_marker.yaml"),
                3,
            )


if __name__ == "__main__":
    unittest.main()
