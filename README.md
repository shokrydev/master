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
  gaia_finetuning_shared.yaml   # optional GAIA configuration

configs/evaluation/
  bigearthnet_txt.yaml          # BEN.txt benchmark prediction export config

scripts/
  download_artifacts.py         # Qwen, SatCLIP and BigEarthNet encoder artifacts
  submit_smoke_job.sh           # short Slurm validation helper
  submit_finetuning_job.sh      # full BEN.txt Slurm submission helper
  finetune_job.sbatch           # single-GPU BEN.txt Slurm job
  submit_evaluation_job.sh      # BEN.txt benchmark export submission helper
  evaluate_job.sbatch           # single-GPU BEN.txt benchmark export job

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

src/evaluation/
  bentxt_records.py             # exported prediction schema loading
  bentxt_parsing.py             # strict BEN.txt answer parsers
  bentxt_scoring.py             # metrics and stratified score tables
  main.py                       # offline scoring CLI
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

Fill in the server-local paths. Directory examples intentionally end in `/`;
file examples do not.

```bash
# Directory of the BigEarthNet-v2 LMDB environment, e.g. BENv2.lmdb/.
BIGEARTHNET_V2_LMDB_ROOT=/absolute/path/to/BENv2.lmdb/
# File path to the BigEarthNet.txt parquet metadata.
BIGEARTHNET_TXT_PARQUET_PATH=/absolute/path/to/BigEarthNet.txt.parquet
# Directory containing, or to be populated with, config.json and model.safetensors.
BIGEARTHNET_ENCODER_DIR=/absolute/path/to/mobilevit_s-all-v0.2.0/
# File path to the SatCLIP checkpoint.
SATCLIP_CHECKPOINT_PATH=/absolute/path/to/satclip-vit16-l10.ckpt
# Hugging Face cache root directory.
HF_HOME=${HOME}/.cache/huggingface/
# Directory under which Slurm run directories are created.
FINETUNING_OUTPUT_ROOT=/absolute/path/to/finetuning_outputs/
# Stable Slurm partition default for this machine. Override per run when needed.
SLURM_DEFAULT_PARTITION=big_job
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

METEOR caption scoring also needs the small WordNet resources used by NLTK:

```bash
uv run python -m nltk.downloader wordnet omw-1.4
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

The Slurm partition comes from `.env` by default. Use command line overrides
for individual submissions instead of editing `.env` between runs:

```bash
./scripts/submit_smoke_job.sh --partition small_job --dry-run
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

For a longer or larger run, make the scheduling choice explicit in the
submission command:

```bash
./scripts/submit_finetuning_job.sh --size 8B --partition big_job --time 7-00:00:00 --dry-run
```

Each Slurm job derives its own output paths:

```text
${FINETUNING_OUTPUT_ROOT}bigearthnet_$SLURM_JOB_ID
${FINETUNING_OUTPUT_ROOT}bigearthnet_$SLURM_JOB_ID/qlora_adapter
```

## Benchmark Evaluation

Final BigEarthNet.txt evaluation is split into GPU prediction export and
offline benchmark scoring. The Slurm evaluation job uses `main.py test` to load
the trained adapter, write `predictions.jsonl` on the `bench` split, and run
the offline scorer to produce metric tables:

```bash
python -m src.evaluation.main score \
  /absolute/path/to/predictions.jsonl \
  --output-dir /absolute/path/to/scored_predictions
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

Direct runs default to `./outputs` and `./outputs/qlora_adapter`. Set
`FINETUNING_OUTPUT_DIR` and `FINETUNING_ADAPTER_DIR` only when the run should
write elsewhere.

## Tests

```bash
uv run python -m unittest discover tests
```
