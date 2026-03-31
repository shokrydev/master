from importlib import import_module

__all__ = [
    "GAIADataset",
    "GAIADataModule",
    "GeoAwareCollator",
]


def __getattr__(name: str):
    if name in {"GAIADataset", "GAIADataModule"}:
        module = import_module("src.data_modules.gaia_datamodule")
        return getattr(module, name)
    if name == "GeoAwareCollator":
        module = import_module("src.data_modules.geo_aware_collator")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
