# Geolocation-Conditioned Remote Sensing VLM

This repository finetunes Qwen3-VL for remote sensing experiments that test
whether explicit geolocation improves VLM performance.

The main experimental configuration uses **BigEarthNet.txt**, which combines
language supervision, geolocation metadata and multispectral/SAR imagery. Each
sample can provide:

- an RGB rendering derived from optical reflectance bands for Qwen3-VL's native
  vision path
- normalized Sentinel-1 SAR and Sentinel-2 multispectral tensors for a frozen
  BigEarthNet MobileViT encoder
- optional geolocation conditioning as part of the prompt text (`loc_text`) or
  as projected SatCLIP tokens (`loc_embed`)

## Repository Map

```text
configs/finetuning/
  bigearthnet_txt_shared.yaml   # primary BEN.txt finetuning config
  bigearthnet_txt_smoke.yaml    # short-run validation override
  loc_text.yaml                 # text-token location conditioning
  loc_embed.yaml                # SatCLIP-token location conditioning
  gaia_finetuning_shared.yaml   # secondary GAIA path

scripts/
  download_artifacts.py         # Qwen, SatCLIP and BigEarthNet encoder artifacts
  submit_smoke_job.sh           # short Slurm validation helper
  submit_finetuning_job.sh      # full BEN.txt Slurm submission helper
  finetune_job.sbatch           # single-GPU BEN.txt Slurm job

src/data_modules/
  ben_txt_datamodule.py         # BigEarthNet.txt loader
  gaia_datamodule.py            # GAIA loader
  geo_aware_collator.py         # shared Qwen/geo/non-RGB collation

src/lightning_modules/
  qwen3_vl_module.py            # QLoRA training, loc and non-RGB conditioning

src/models/
  bigearthnet_s1s2_encoder.py
  non_rgb_modality_projection.py
  location_modality_projection.py
  satclip/
```

## Environment

Reproduce the Python environment from `pyproject.toml` with `uv`:

```bash
uv sync
```

Create a private repo-root `.env` from the template:

```bash
cp .env.example .env
```

Fill in the server-local paths:

```bash
BIGEARTHNET_V2_LMDB_ROOT=/path/to/BigEarthNet-v2-lmdb
BIGEARTHNET_TXT_PARQUET_PATH=/path/to/BigEarthNet.txt.parquet
BIGEARTHNET_ENCODER_DIR=/path/to/mobilevit_s-all-v0.2.0
SATCLIP_CHECKPOINT_PATH=/path/to/satclip-vit16-l10.ckpt
HF_HOME=/path/to/huggingface-cache
FINETUNING_OUTPUT_ROOT=/path/to/finetuning_outputs
```

The real `.env` is machine-specific and intentionally ignored by version
control; `.env.example` documents the required variables.

## BigEarthNet.txt Data

This repository does not prepare BigEarthNet.txt itself. Use the
[official Hugging Face dataset instructions](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
to prepare the data, then set the corresponding LMDB and parquet paths in
`.env`.

Expected local inputs:

- BigEarthNet-v2 imagery converted to the LMDB layout used by the datamodule
- BigEarthNet.txt parquet metadata

## Model Artifacts

The artifact helper reads `.env` and downloads the model files needed for the
default initial run:

```bash
uv run python scripts/download_artifacts.py --dry-run
uv run python scripts/download_artifacts.py
```

Default downloads:

- `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit`
- `microsoft/SatCLIP-ViT16-L10`
- `BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.2.0`

To prefetch all Qwen model sizes:

```bash
uv run python scripts/download_artifacts.py --all
```

The BigEarthNet MobileViT wrapper loads `config.json` and
`model.safetensors` directly through `timm` and `safetensors`.

## Setup Check

Before submitting a Slurm job, validate the local paths, required artifacts and
BigEarthNet `loc_embed` smoke config:

```bash
uv run python scripts/check_server_setup.py
```

## Slurm Submission

Use `submit_smoke_job.sh` for a short Slurm validation run before launching a
full finetuning job. Its default checks the full architecture with:

- Qwen3-VL 2B
- BigEarthNet.txt
- enabled non-RGB S1/S2 conditioning
- `loc_embed`
- short-run trainer settings

```bash
# Print the sbatch command and resolved paths without submitting.
./scripts/submit_smoke_job.sh --dry-run

# Submit the short validation run.
./scripts/submit_smoke_job.sh
```

Other condition checks:

```bash
# Print alternative condition submissions without launching jobs.
./scripts/submit_smoke_job.sh --condition no_loc --dry-run
./scripts/submit_smoke_job.sh --condition loc_text --dry-run
```

Use `submit_finetuning_job.sh` for full runs. Its default is the full
BigEarthNet.txt 2B `loc_embed` run:

```bash
./scripts/submit_finetuning_job.sh --dry-run
./scripts/submit_finetuning_job.sh
```

Each Slurm job derives its own output paths:

```text
$FINETUNING_OUTPUT_ROOT/bigearthnet_$SLURM_JOB_ID
$FINETUNING_OUTPUT_ROOT/bigearthnet_$SLURM_JOB_ID/adapter
```

## Direct Local Command

For an interactive GPU shell without Slurm:

```bash
set -a
source .env
set +a

uv run python main.py fit \
  --config configs/finetuning/bigearthnet_txt_shared.yaml \
  --config configs/finetuning/loc_embed.yaml \
  --config configs/finetuning/bigearthnet_txt_smoke.yaml \
  --trainer.devices 1
```

Set `FINETUNING_OUTPUT_DIR` and `FINETUNING_ADAPTER_DIR` manually for direct
runs if you do not use the Slurm helper.

## Tests

```bash
uv run python -m unittest discover tests
```
