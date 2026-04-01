from importlib import import_module

__all__ = [
    "BIGEARTHNET_19_ALIASES",
    "BIGEARTHNET_19_LABELS",
    "BigEarthNetMultilabelEvaluator",
]


def __getattr__(name: str):
    if name in {
        "BIGEARTHNET_19_ALIASES",
        "BIGEARTHNET_19_LABELS",
        "BigEarthNetMultilabelEvaluator",
    }:
        module = import_module("src.evaluation.bigearthnet_templated_multilabel")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
