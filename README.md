# Repository Structure

```text
master/
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── checkpoints/
├── configs/
│   ├── finetuning/
│   ├── evaluation/
│   └── ...
├── logs/
├── notebooks/
├── src/
│   ├── callbacks/
│   │   └── __init__.py
│   ├── data_modules/
│   │   ├── __init__.py
│   │   ├── ben_txt_datamodule.py
│   │   ├── gaia_datamodule.py
│   │   └── geo_aware_collator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── bigearthnet_templated_multilabel.py
│   ├── lightning_modules/
│   │   ├── __init__.py
│   │   └── qwen3_vl_module.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── captioning.py
│   │   ├── multilabel_classification.py
│   │   └── vqa.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── location_modality_projection.py
│   │   └── satclip/
│   │       ├── ...
│   └── utils/
│       ├── __init__.py
│       └── continent_lookup.py
├── scripts/
│   ├── download_satclip.py
│   ├── evaluate_runs.py
│   ├── finetune_job.sbatch
│   ├── prefetch_qwen3vl_weights.sh
│   └── submit_finetuning_job.sh
└── tests/
    ├── test_gaia_datamodule.py
    ├── test_loc_embed.py
    └── test_save_qlora_adapters.py
```

# Required External Assets

This repository does not track large model or dataset artifacts. You must provide them externally.

## 1) Unsloth Qwen3-VL 4-bit model repos

Used for the 2B / 4B / 8B runs:

- `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit`
- `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`
- `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`

Prefetch into your HF cache (recommended before cluster jobs):

```bash
bash scripts/prefetch_qwen3vl_weights.sh 2B 4B 8B
```

## 2) BigEarthNet.txt + BigEarthNet-v2 assets

Reference:
- BigEarthNet.txt dataset card: `https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`

Expected preparation flow:
- download BigEarthNet-v2 imagery
- convert imagery to LMDB
- obtain the `BigEarthNet.txt` parquet metadata from the dataset card
- set both env vars:

```bash
export BIGEARTHNET_V2_LMDB_ROOT=/data/datasets/BigEarthNet-V2
export BIGEARTHNET_TXT_PARQUET_PATH=/data/<your_own_directory>/BigEarthNet.txt.parquet
```

## 3) SatCLIP checkpoint (loc_embed only)

`loc_embed` requires a [SatCLIP](https://github.com/microsoft/satclip) checkpoint (recommended variant: `SatCLIP-ViT16-L10`).

```bash
python scripts/download_satclip.py --model SatCLIP-ViT16-L10 --output_dir /path/to/satclip
export SATCLIP_CHECKPOINT_PATH=/path/to/satclip/satclip-vit16-l10.ckpt
```

## 4) GAIA dataset root

Set up [GAIA](https://github.com/Orion-AI-Lab/GAIA) following the repository instructions, then point the code to the local GAIA root:

```bash
export GAIA_ROOT=/path/to/GAIA
```

Expected under that root:

- `train/`
- `val/`
- `train_data.json`
- `val_data.json`
