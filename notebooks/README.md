# Notebooks

This directory is for local experiment inspection and thesis-figure preparation.
Reusable parsing and plotting code belongs in `notebooks/utils/`; notebooks
should call those helpers rather than duplicating logic in cells.

Typical workflow:

```bash
# After SERVER_SYNC_HOST, SERVER_REPO_ROOT and SERVER_FINETUNING_OUTPUT_ROOT
# are set in .env, sync currently active Slurm jobs:
./scripts/sync_server_runs.sh --jobs-from-squeue

# Or sync all job ids recorded in planning/run_registry.md:
./scripts/sync_server_runs.sh

uv run python notebooks/utils/extract_training_curves.py \
  --jobs 11270 11271 11272 \
  --tags train/loss val/loss

uv run python notebooks/utils/plot_training_curves.py \
  --jobs 11270 11271 11272 \
  --tag val/loss \
  --output outputs/analysis/figures/2b_val_loss.png
```

Generated CSVs and figures go under `outputs/analysis/` and are ignored by Git.
