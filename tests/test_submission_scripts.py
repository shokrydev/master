import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class SubmissionScriptsTest(unittest.TestCase):
    @staticmethod
    def _write_server_env(workdir: Path) -> None:
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

    def test_early_pipeline_submits_correct_and_shuffled_evaluations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                'if [ "${1:-}" = "--parsable" ]; then\n'
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

    def test_full_trajectory_matrix_submits_all_fits_before_evaluations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_2b_full_trajectory_matrix.sh",
                "submit_finetuning_job.sh",
                "submit_evaluation_job.sh",
            ):
                source = REPO_ROOT / "scripts" / name
                target = scripts_dir / name
                shutil.copy2(source, target)

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            counter_path = workdir / "sbatch-counter"
            counter_path.write_text("90000\n", encoding="utf-8")
            call_log = workdir / "sbatch-calls"
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                f"counter_file={counter_path!s}\n"
                f"call_log={call_log!s}\n"
                'counter=$(cat "$counter_file")\n'
                "counter=$((counter + 1))\n"
                'echo "$counter" > "$counter_file"\n'
                'printf \'%s\\n\' "$*" >> "$call_log"\n'
                'if [ "${1:-}" = "--parsable" ]; then\n'
                '    echo "$counter"\n'
                "else\n"
                '    echo "Submitted batch job $counter"\n'
                "fi\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [str(scripts_dir / "submit_2b_full_trajectory_matrix.sh")],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 68)
            self.assertTrue(all("--parsable" in call for call in calls[:8]))
            self.assertTrue(all("--parsable" not in call for call in calls[8:]))
            self.assertEqual(result.stdout.count("Submitted batch job"), 68)
            self.assertIn(
                "/tmp/finetuning/bigearthnet_90001/qlora_adapter_steps/step_000050",
                result.stdout,
            )
            self.assertIn(
                "/tmp/finetuning/bigearthnet_90008/qlora_adapter",
                result.stdout,
            )
            self.assertEqual(
                result.stdout.count("--data.init_args.coordinate_perturbation shuffled"),
                12,
            )

    def test_evaluation_batch_profile_submission_preserves_dependency_and_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_evaluation_batch_profile_job.sh",
                "profile_bentxt_evaluation_batch_size.sbatch",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\nprintf '%s\\n' \"$*\"\necho 'Submitted batch job 99991'\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_evaluation_batch_profile_job.sh"),
                    "--adapter-dir",
                    "/tmp/future-adapter",
                    "--dependency",
                    "afterok:11807",
                    "--batch-sizes",
                    "16",
                    "64",
                    "256",
                    "--memory-safety-fraction",
                    "0.90",
                    "--worker-counts",
                    "8",
                    "10",
                    "12",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--dependency=afterok:11807", result.stdout)
            self.assertIn("--batch-sizes 16 64 256", result.stdout)
            self.assertIn("--memory-safety-fraction 0.90", result.stdout)
            self.assertIn("--worker-counts 8 10 12", result.stdout)
            self.assertIn("Submitted batch job 99991", result.stdout)


if __name__ == "__main__":
    unittest.main()
