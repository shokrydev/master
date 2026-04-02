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
│   ├── data/
│   ├── finetuning/
│   ├── evaluation/
│   └── model/
├── data/
│   ├── processed/
│   └── raw/
├── logs/
├── notebooks/
├── src/
│   ├── callbacks/
│   │   └── __init__.py
│   ├── data_modules/
│   │   ├── __init__.py
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
└── tests/
    └── test_loc_embed.py
```
