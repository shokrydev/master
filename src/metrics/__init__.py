from importlib import import_module

__all__ = [
    "CaptioningMetrics",
    "VQAAccuracy",
    "VQAAccuracyMultiRef",
    "ExactMatchAccuracy",
    "TokenF1Score",
    "MultiLabelClassificationMetrics",
    "MeanAveragePrecision",
]


def __getattr__(name: str):
    if name == "CaptioningMetrics":
        module = import_module("src.metrics.captioning")
        return getattr(module, name)
    if name in {"MeanAveragePrecision", "MultiLabelClassificationMetrics"}:
        module = import_module("src.metrics.multilabel_classification")
        return getattr(module, name)
    if name in {"ExactMatchAccuracy", "TokenF1Score", "VQAAccuracy", "VQAAccuracyMultiRef"}:
        module = import_module("src.metrics.vqa")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
