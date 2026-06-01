# Captioning metrics: BLEU (1-4), ROUGE-1, ROUGE-2, ROUGE-L, METEOR, CIDEr
# Uses pycocoevalcap for BLEU/CIDEr, nltk for METEOR, rouge-score for ROUGE variants


import nltk
import torch
from nltk.translate.meteor_score import meteor_score
from torchmetrics import Metric


class CaptioningMetrics(Metric):
    """Corpus-level captioning metrics with multi-reference support.

    Computes BLEU (1-4), ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and CIDEr.
    Accumulates predictions/references across batches and computes
    corpus-level scores at epoch end.

    Usage:
        metric = CaptioningMetrics()
        metric.update(["a cat on a mat"], [["a cat on a mat", "cat sitting on mat"]])
        scores = metric.compute()  # {"bleu1": ..., "rouge_1": ..., "cider": ..., ...}
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_meteor_resources()
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("references", default=[], dist_reduce_fx=None)

    def update(self, predictions: list[str], references: list[list[str]]) -> None:
        """Accumulate predictions and their multi-reference ground truths.

        Args:
            predictions: List of predicted caption strings (length B).
            references: List of reference caption lists (length B), each inner
                list contains one or more reference strings.
        """
        self.predictions.extend(predictions)
        self.references.extend(references)

    def _compute_rouge(self) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L using rouge-score.

        For multi-reference, takes the max F-score across references per example,
        then averages across all examples.
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = len(self.predictions)

        for pred, refs in zip(self.predictions, self.references, strict=True):
            # Max over references for each ROUGE variant
            best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            for ref in refs:
                scores = scorer.score(ref, pred)
                for key in best:
                    best[key] = max(best[key], scores[key].fmeasure)
            for key in sums:
                sums[key] += best[key]

        return {k: v / n for k, v in sums.items()} if n > 0 else sums

    def _ensure_meteor_resources(self) -> None:
        """Check METEOR resources once when caption metrics are initialized."""
        missing = []
        for resource in ("corpora/wordnet", "corpora/omw-1.4"):
            try:
                nltk.data.find(resource)
            except LookupError:
                missing.append(resource)
        if missing:
            raise RuntimeError(
                "METEOR requires NLTK WordNet data. Install it once with: "
                "uv run python -m nltk.downloader wordnet omw-1.4"
            )

    def _compute_meteor(self) -> float:
        """Compute METEOR using nltk (pure Python, no Java dependency).

        For multi-reference, nltk.meteor_score picks the best reference automatically.
        Returns the corpus-level average.
        """
        total = 0.0
        n = len(self.predictions)
        for pred, refs in zip(self.predictions, self.references, strict=True):
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_pred = pred.split()
            total += meteor_score(tokenized_refs, tokenized_pred)
        return total / n if n > 0 else 0.0

    def compute(self) -> dict[str, torch.Tensor]:
        """Run all scorers on accumulated data."""
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider

        metric_names = [
            "bleu1", "bleu2", "bleu3", "bleu4",
            "rouge_1", "rouge_2", "rouge_l", "meteor", "cider",
        ]

        if len(self.predictions) == 0:
            return {k: torch.tensor(0.0) for k in metric_names}

        # Build COCO-style dicts for pycocoevalcap: {id: [caption, ...]}
        gts = {}
        res = {}
        for i, (pred, refs) in enumerate(zip(self.predictions, self.references, strict=True)):
            res[i] = [pred]
            gts[i] = [str(r) for r in refs]

        results = {}

        # BLEU 1-4 (pycocoevalcap, corpus-level with multi-reference)
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        for name, s in zip(["bleu1", "bleu2", "bleu3", "bleu4"], bleu_scores, strict=True):
            results[name] = torch.tensor(float(s))

        # METEOR (nltk, pure Python — no Java subprocess)
        results["meteor"] = torch.tensor(self._compute_meteor())

        # CIDEr (pycocoevalcap)
        cider_score, _ = Cider().compute_score(gts, res)
        results["cider"] = torch.tensor(float(cider_score))

        # ROUGE-1, ROUGE-2, ROUGE-L (rouge-score, max over references)
        rouge_scores = self._compute_rouge()
        results["rouge_1"] = torch.tensor(rouge_scores["rouge1"])
        results["rouge_2"] = torch.tensor(rouge_scores["rouge2"])
        results["rouge_l"] = torch.tensor(rouge_scores["rougeL"])

        return results

    def reset(self) -> None:
        """Clear accumulated predictions and references."""
        self.predictions = []
        self.references = []
