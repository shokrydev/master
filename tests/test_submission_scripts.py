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

    @staticmethod
    def _write_fit_adapter_dirs(workdir: Path, fit_job: str) -> None:
        finetuning_root = workdir / "finetuning"
        with (workdir / ".env").open("a", encoding="utf-8") as handle:
            handle.write(f"FINETUNING_OUTPUT_ROOT={finetuning_root}\n")
        run_root = finetuning_root / f"bigearthnet_{fit_job}"
        (run_root / "qlora_adapter").mkdir(parents=True)
        for step in (50, 100, 500, 1000, 5000):
            (run_root / "qlora_adapter_steps" / f"step_{step:06d}").mkdir(parents=True)

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
                    "--size",
                    "4B",
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
            self.assertIn("--size 4B", result.stdout)
            self.assertIn("Model size: 4B", result.stdout)
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

    def test_after_marker_ablation_submits_two_fits_and_complete_dependent_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            for name in (
                "submit_loc_embed_after_marker_ablation.sh",
                "submit_finetuning_job.sh",
                "submit_2b_trajectory_evaluations.sh",
                "submit_evaluation_job.sh",
            ):
                shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)
            config_dir = workdir / "configs/finetuning/ablations"
            config_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "configs/finetuning/ablations/loc_embed_after_marker.yaml",
                config_dir / "loc_embed_after_marker.yaml",
            )

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            counter = workdir / "counter"
            counter.write_text("60000\n", encoding="utf-8")
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
                'if [ "${1:-}" = "--parsable" ]; then\n'
                '    echo "$job"\n'
                "else\n"
                '    echo "Submitted batch job $job"\n'
                "fi\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"

            result = subprocess.run(
                [str(scripts_dir / "submit_loc_embed_after_marker_ablation.sh")],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 20)
            self.assertTrue(all("finetune_job.sbatch" in call for call in calls[:2]))
            self.assertIn("--seed_everything 42", calls[0])
            self.assertIn("--seed_everything 43", calls[1])
            configured_calls = calls[:10] + calls[11:19]
            self.assertTrue(
                all("loc_embed_after_marker.yaml" in call for call in configured_calls)
            )
            seed42_evaluations = calls[2:10]
            seed43_evaluations = calls[11:19]
            self.assertTrue(
                all("--dependency=afterok:60001" in call for call in seed42_evaluations)
            )
            self.assertTrue(
                all("--dependency=afterok:60002" in call for call in seed43_evaluations)
            )
            self.assertIn(
                "--dependency=afterok:60003:60004:60005:60006:60007:60008:60009:60010",
                calls[10],
            )
            self.assertIn(
                "--dependency=afterok:60012:60013:60014:60015:60016:60017:60018:60019",
                calls[19],
            )
            self.assertEqual(
                sum("--data.init_args.coordinate_perturbation shuffled" in call for call in calls),
                4,
            )
            self.assertIn(
                "Submitted 2 fits, 16 dependent evaluations and 2 dependent fit-level CLAIR jobs.",
                result.stdout,
            )

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
            self._write_fit_adapter_dirs(workdir, "11881")
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

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            counter = workdir / "counter"
            counter.write_text("50000\n", encoding="utf-8")
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
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"

            for fit_job, (condition, seed, count) in expected.items():
                self._write_fit_adapter_dirs(workdir, fit_job)
                manifest = workdir / f"trajectory_{fit_job}.tsv"
                result = subprocess.run(
                    [
                        str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                        "--fit-job",
                        fit_job,
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
                self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), count + 1)
                self.assertIn(
                    f"Submitted {count} trajectory evaluation jobs for fit {fit_job} "
                    f"({condition}, seed {seed}).",
                    result.stdout,
                )
                self.assertIn(f"Submitted one CLAIR job for fit {fit_job}.", result.stdout)

            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 34)
            evaluation_calls = [call for call in calls if "evaluate_job.sbatch" in call]
            self.assertEqual(len(evaluation_calls), 30)
            self.assertTrue(all("afterok:118" not in call for call in evaluation_calls))

    def test_fit_level_clair_uses_syncable_job_directory(self):
        script = (REPO_ROOT / "scripts/score_clair_job.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('run_output="${EVALUATION_OUTPUT_ROOT%/}/clair_${SLURM_JOB_ID}"', script)
        self.assertNotIn("/clair_fit_", script)

    def test_packed_trajectory_helper_dry_run_plans_one_job_for_four_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(
                REPO_ROOT / "scripts/submit_fit_trajectory_evaluation_job.sh",
                scripts_dir,
            )

            result = subprocess.run(
                [
                    str(scripts_dir / "submit_fit_trajectory_evaluation_job.sh"),
                    "--fit-job",
                    "11809",
                    "--condition",
                    "loc_embed",
                    "--size",
                    "2B",
                    "--seed",
                    "42",
                    "--correct-step",
                    "250",
                    "--correct-step",
                    "2500",
                    "--correct-step",
                    "10000",
                    "--correct-step",
                    "20000",
                    "--dry-run",
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Entries: 4", result.stdout)
            self.assertEqual(result.stdout.count("evaluate_trajectory_job.sbatch"), 1)
            self.assertIn("step_000250_correct", result.stdout)
            self.assertIn("step_020000_correct", result.stdout)
            self.assertIn("[Dry run - not writing or submitting]", result.stdout)
            self.assertFalse((workdir / "outputs").exists())

    def test_packed_trajectory_helper_submits_one_parent_and_one_clair_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(
                REPO_ROOT / "scripts/submit_fit_trajectory_evaluation_job.sh",
                scripts_dir,
            )
            finetuning_root = workdir / "finetuning"
            with (workdir / ".env").open("a", encoding="utf-8") as handle:
                handle.write(f"FINETUNING_OUTPUT_ROOT={finetuning_root}\n")
            fit_root = finetuning_root / "bigearthnet_13000"
            (fit_root / "qlora_adapter").mkdir(parents=True)
            for step in (50, 100, 500, 1000, 5000):
                (fit_root / "qlora_adapter_steps" / f"step_{step:06d}").mkdir(
                    parents=True
                )

            mock_bin = workdir / "bin"
            mock_bin.mkdir()
            call_log = workdir / "sbatch-calls"
            counter = workdir / "counter"
            counter.write_text("61000\n", encoding="utf-8")
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                "#!/bin/bash\n"
                f"call_log={call_log!s}\n"
                f"counter={counter!s}\n"
                'printf \'%s\\n\' "$*" >> "$call_log"\n'
                'job=$(cat "$counter")\n'
                'job=$((job + 1))\n'
                'echo "$job" > "$counter"\n'
                'if [ "${1:-}" = "--parsable" ]; then\n'
                '    echo "$job"\n'
                "else\n"
                '    echo "Submitted batch job $job"\n'
                "fi\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)
            manifest = workdir / "packed.tsv"
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"

            result = subprocess.run(
                [
                    str(scripts_dir / "submit_fit_trajectory_evaluation_job.sh"),
                    "--fit-job",
                    "13000",
                    "--condition",
                    "loc_text",
                    "--size",
                    "4B",
                    "--seed",
                    "42",
                    "--short-batch",
                    "128",
                    "--bbox-batch",
                    "256",
                    "--caption-batch",
                    "192",
                    "--correct-step",
                    "50",
                    "--correct-step",
                    "100",
                    "--correct-step",
                    "500",
                    "--correct-step",
                    "1000",
                    "--correct-step",
                    "5000",
                    "--correct-step",
                    "final",
                    "--shuffled-step",
                    "1000",
                    "--shuffled-step",
                    "final",
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
            self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 9)
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 2)
            self.assertIn("evaluate_trajectory_job.sbatch", calls[0])
            self.assertIn("--dependency=afterok:61001", calls[1])
            self.assertIn("CLAIR_EXPECTED_EXPORTS=8", calls[1])
            jobs_manifest = workdir / "packed_jobs.tsv"
            self.assertTrue(jobs_manifest.is_file())
            self.assertIn("61001\t61001\t13000\tloc_text\t4B", jobs_manifest.read_text())

    def test_packed_trajectory_helper_refuses_unprofiled_larger_model_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(
                REPO_ROOT / "scripts/submit_fit_trajectory_evaluation_job.sh",
                scripts_dir,
            )

            result = subprocess.run(
                [
                    str(scripts_dir / "submit_fit_trajectory_evaluation_job.sh"),
                    "--fit-job",
                    "13000",
                    "--condition",
                    "loc_embed",
                    "--size",
                    "4B",
                    "--seed",
                    "42",
                    "--correct-step",
                    "final",
                    "--dry-run",
                ],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires profiler-selected", result.stdout)
            self.assertIn("2B values are intentionally not reused", result.stdout)

    def test_packed_trajectory_runner_resumes_completed_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self._write_server_env(workdir)
            evaluation_root = workdir / "evaluation"
            with (workdir / ".env").open("a", encoding="utf-8") as handle:
                handle.write(f"EVALUATION_OUTPUT_ROOT={evaluation_root}\n")
            scripts_dir = workdir / "scripts"
            scripts_dir.mkdir()
            shutil.copy2(
                REPO_ROOT / "scripts/evaluate_trajectory_job.sbatch",
                scripts_dir,
            )
            call_log = workdir / "evaluation-calls"
            evaluator = scripts_dir / "evaluate_job.sbatch"
            evaluator.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\n' \"$EVAL_OUTPUT_DIR\" >> {call_log!s}\n"
                'mkdir -p "$EVAL_OUTPUT_DIR/scored_predictions"\n'
                'printf \'{}\\n\' > "$EVAL_OUTPUT_DIR/predictions.jsonl"\n'
                'printf \'{}\\n\' > "$EVAL_OUTPUT_DIR/scored_predictions/summary.json"\n',
                encoding="utf-8",
            )
            evaluator.chmod(0o755)
            adapter_root = workdir / "adapters"
            adapter_50 = adapter_root / "step_000050"
            adapter_100 = adapter_root / "step_000100"
            adapter_50.mkdir(parents=True)
            adapter_100.mkdir(parents=True)
            plan = workdir / "plan.tsv"
            plan.write_text(
                "evaluation_job\tfit_job\tcondition\tseed\tstep\tcoordinate_setting\tadapter_dir\trun_label\n"
                f"SLURM_JOB_ID/entries/step_000050_correct\t13000\tno_loc\t42\t50\tcorrect\t{adapter_50!s}\trun-50\n"
                f"SLURM_JOB_ID/entries/step_000100_correct\t13000\tno_loc\t42\t100\tcorrect\t{adapter_100!s}\trun-100\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_SUBMIT_DIR": str(workdir),
                    "SLURM_JOB_ID": "62001",
                    "TRAJECTORY_MODEL_SIZE": "2B",
                }
            )

            command = [str(scripts_dir / "evaluate_trajectory_job.sbatch"), str(plan)]
            first = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            env["SLURM_JOB_ID"] = "62002"
            env["TRAJECTORY_OUTPUT_ID"] = "62001"
            second = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(second.stdout.count("Skipping completed entry"), 2)
            resolved = evaluation_root / "trajectory_62001/trajectory_manifest.tsv"
            self.assertIn("62001/entries/step_000050_correct", resolved.read_text())

    def test_completed_fit_helper_refuses_missing_adapters_before_submission(self):
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
            marker = workdir / "sbatch-was-called"
            mock_sbatch = mock_bin / "sbatch"
            mock_sbatch.write_text(
                f"#!/bin/bash\ntouch {marker!s}\nexit 1\n",
                encoding="utf-8",
            )
            mock_sbatch.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{mock_bin}:{env['PATH']}"
            result = subprocess.run(
                [
                    str(scripts_dir / "submit_2b_trajectory_evaluations.sh"),
                    "--fit-job",
                    "11809",
                ],
                cwd=workdir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required adapter directories are missing", result.stdout)
            self.assertIn("No jobs were submitted", result.stdout)
            self.assertFalse(marker.exists())

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
            self._write_fit_adapter_dirs(workdir, "11809")
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
