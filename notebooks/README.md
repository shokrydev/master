# Notebooks

This directory is for local experiment inspection and thesis-figure preparation.
Reusable parsing and plotting code belongs in `notebooks/utils/`; notebooks
should call those helpers rather than duplicating logic in cells.

Synced server evidence is grouped by run type and Slurm job id:

- `outputs/finetuning/<job>/`: copied finetuning logs and lightweight Lightning
  files, including TensorBoard events and resolved configuration.
- `outputs/evaluation/<job>/`: copied evaluation logs, prediction exports and
  score summaries.

Large adapter/checkpoint files are intentionally not synced.

Available notebooks:

- `training_diagnostics.ipynb`: compares synced TensorBoard loss curves and
  qualitative generations.
- `benchmark_2b_core.ipynb`: two-seed full-budget 2B core comparison plus the
  selected additive-SatCLIP extension and matched shuffled-coordinate controls.
- `benchmark_4b_core.ipynb`: complete 4B core comparison with matched
  shuffled-coordinate controls.
- `benchmark_uncertainty.ipynb`: paired patch-cluster bootstrap intervals for
  core and shuffled-coordinate benchmark contrasts across model sizes.
- `ablation_2b_1000_steps.ipynb`: comprehensive repaired-placement 2B
  1000-step development comparison, including weaker controls, shuffled
  coordinates, architecture ablations and cross-seed replication.
- `archive/benchmark_2b_historical.ipynb`: superseded historical-placement
  benchmark retained for debugging and provenance.

Typical workflow:

```bash
# After the SERVER_SYNC_* variables are set in .env, sync currently active
# Slurm jobs:
./scripts/sync_server_runs.sh --jobs-from-squeue

# Or sync all job ids recorded in planning/run_registry.md:
./scripts/sync_server_runs.sh

uv run python -m notebooks.utils.training_curves \
  --jobs 11270 11271 11272 \
  --tag val/loss \
  --output notebooks/analysis/figures/2b_val_loss.png
```

Generated CSVs and figures go under `notebooks/analysis/` and are ignored by Git.
