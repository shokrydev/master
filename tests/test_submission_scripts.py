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
            self.assertTrue(
                all(
                    "--seed_everything 42 --data.init_args.training_shuffle_seed 42"
                    in call
                    for call in calls[:4]
                )
            )
            self.assertTrue(
                all(
                    "--seed_everything 43 --data.init_args.training_shuffle_seed 43"
                    in call
                    for call in calls[4:8]
                )
            )
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

    def test_existing_fit_trajectory_helper_dry_run_submits_only_evaluations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_2b_trajectory_evaluations.sh",
                "submit_evaluation_job.sh",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)

            result = subprocess.run(
                [
                    str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                    "--fit-job",
                    "11809",
                    "--dry-run",
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("[Dry run - not submitting]"), 8)
            self.assertEqual(result.stdout.count("coordinate_perturbation shuffled"), 2)
            self.assertIn("bigearthnet_11809/qlora_adapter_steps/step_000050", result.stdout)
            self.assertIn("bigearthnet_11809/qlora_adapter", result.stdout)
            self.assertIn("evaluation_batch_sizes.short_answer 256", result.stdout)
            self.assertIn("evaluation_batch_sizes.bounding_box 512", result.stdout)
            self.assertIn("evaluation_batch_sizes.captioning 384", result.stdout)
            self.assertIn("evaluation_num_workers_by_bucket.short_answer 8", result.stdout)
            self.assertNotIn("submit_finetuning_job", result.stdout)

    def test_existing_fit_trajectory_helper_writes_job_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_2b_trajectory_evaluations.sh",
                "submit_evaluation_job.sh",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            counter = workdir / "counter"
            counter.write_text("30000\n", encoding="utf-8")
            call_log = workdir / "sbatch-calls"
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                f"counter={counter!s}\n"
                f"call_log={call_log!s}\n"
                'printf \'%s\\n\' "$*" >> "$call_log"\n'
                'job=$(cat "$counter")\n'
                'job=$((job + 1))\n'
                'echo "$job" > "$counter"\n'
                'echo "Submitted batch job $job"\n',
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)
            manifest = workdir / "trajectory.tsv"
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                    "--short-batch",
                    "128",
                    "--bbox-batch",
                    "64",
                    "--caption-batch",
                    "16",
                    "--short-workers",
                    "12",
                    "--bbox-workers",
                    "10",
                    "--caption-workers",
                    "8",
                    "--fit-job",
                    "11881",
                    "--manifest",
                    str(manifest),
                    "--submit-clair",
                    "--clair-model",
                    "unsloth/test-judge",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 7)
            self.assertTrue(lines[1].startswith("30001\t11881\tno_loc\t42\t50\tcorrect\t"))
            self.assertIn("\t11881\tno_loc\t42\tfinal\tcorrect\t", lines[-1])
            clair_lines = (workdir / "trajectory_clair_jobs.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(clair_lines), 2)
            self.assertTrue(clair_lines[1].startswith("30007\t11881\tno_loc\t42\t"))
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 7)
            self.assertIn("--dependency=afterok:30001:30002:30003:30004:30005:30006", calls[6])
            self.assertIn("CLAIR_FIT_JOB=11881", calls[-1])
            self.assertIn("--batch-size 64 --max-new-tokens 512", calls[-1])

    def test_completed_fit_mappings_submit_expected_evaluation_counts(self):
        expected = {
            "11809": ("loc_embed", "42", 8),
            "11810": ("loc_additive_satclip", "42", 8),
            "11811": ("no_loc", "43", 6),
            "11812": ("loc_text", "43", 8),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_2b_trajectory_evaluations.sh",
                "submit_evaluation_job.sh",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)

            for fit_job, (condition, seed, count) in expected.items():
                result = subprocess.run(
                    [
                        str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                        "--fit-job",
                        fit_job,
                        "--submit-clair",
                        "--dry-run",
                    ],
                    cwd=workdir,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count("[Dry run - not submitting]"), count)
                self.assertIn(
                    f"Submitted {count} trajectory evaluation jobs for fit {fit_job} "
                    f"({condition}, seed {seed}).",
                    result.stdout,
                )
                self.assertIn(
                    f"A real submission would add one CLAIR job for fit {fit_job}.",
                    result.stdout,
                )

    def test_fit_level_clair_uses_syncable_job_directory(self):
        script = (REPO_ROOT / "scripts/score_clair_job.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('run_output="${EVALUATION_OUTPUT_ROOT%/}/clair_${SLURM_JOB_ID}"', script)
        self.assertNotIn("/clair_fit_", script)

    def test_location_fit_manifest_and_clair_dependency_include_all_eight_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_2b_trajectory_evaluations.sh",
                "submit_evaluation_job.sh",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)
            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            counter = workdir / "counter"
            counter.write_text("40000\n", encoding="utf-8")
            call_log = workdir / "sbatch-calls"
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                f"counter={counter!s}\n"
                f"call_log={call_log!s}\n"
                'printf \'%s\\n\' "$*" >> "$call_log"\n'
                'job=$(cat "$counter")\n'
                'job=$((job + 1))\n'
                'echo "$job" > "$counter"\n'
                'echo "Submitted batch job $job"\n',
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)
            manifest = workdir / "loc_embed.tsv"
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                    "--fit-job",
                    "11809",
                    "--manifest",
                    str(manifest),
                    "--submit-clair",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = manifest.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 9)
            self.assertEqual(sum("\tshuffled\t" in row for row in rows), 2)
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 9)
            self.assertIn(
                "--dependency=afterok:40001:40002:40003:40004:40005:40006:40007:40008",
                calls[-1],
            )

    def test_clair_submission_exports_model_and_forwards_scorer_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(REPO_ROOT / "scripts/submit_clair_job.sh", scripts_dir)
            predictions = workdir / "predictions.jsonl"
            predictions.touch()
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_clair_job.sh"),
                    "--predictions",
                    str(predictions),
                    "--model",
                    "unsloth/test-judge",
                    "--batch-size",
                    "8",
                    "--limit",
                    "32",
                    "--dry-run",
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CLAIR_MODEL_NAME_OR_PATH=unsloth/test-judge", result.stdout)
            self.assertIn("--batch-size 8 --limit 32", result.stdout)
            self.assertIn("[Dry run - not submitting]", result.stdout)

    def test_clair_batch_profile_submission_forwards_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(REPO_ROOT / "scripts/submit_clair_batch_profile_job.sh", scripts_dir)
            predictions = workdir / "predictions.jsonl"
            predictions.touch()
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_clair_batch_profile_job.sh"),
                    "--predictions",
                    str(predictions),
                    "--batch-sizes",
                    "8",
                    "16",
                    "24",
                    "--profile-rows",
                    "48",
                    "--dry-run",
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("profile_clair_batch_size.sbatch", result.stdout)
            self.assertIn("--batch-sizes 8 16 24 --profile-rows 48", result.stdout)
            self.assertIn("[Dry run - not submitting]", result.stdout)


if __name__ == "__main__":
    unittest.main()
