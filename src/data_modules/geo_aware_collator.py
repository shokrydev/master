# Geo-aware collator for VLM data pipelines
# Wraps Unsloth's collator to pass through lat/lon tensors

from typing import Any, Dict, List

import torch


class GeoAwareCollator:
    """Wraps an inner collator (e.g. UnslothVisionDataCollator) to handle geo fields.

    Before inner collation: strips lat, lon, image_id from items (the inner
    collator only understands 'messages').
    After inner collation: re-attaches lat/lon as stacked float64 tensors.
    """

    def __init__(self, inner_collator, has_geo: bool = False):
        self.inner_collator = inner_collator
        self.has_geo = has_geo

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Strip extra fields before passing to inner collator
        lats, lons, image_ids, references = [], [], [], []
        cleaned = []
        for item in items:
            item = dict(item)  # shallow copy
            image_ids.append(item.pop("image_id", None))
            references.append(item.pop("references", None))
            if self.has_geo:
                lats.append(item.pop("lat", None))
                lons.append(item.pop("lon", None))
            else:
                item.pop("lat", None)
                item.pop("lon", None)
            cleaned.append(item)

        # Inner collation (produces input_ids, attention_mask, labels, pixel_values, etc.)
        batch = self.inner_collator(cleaned)

        # Re-attach geo tensors
        if self.has_geo and lats and lats[0] is not None:
            batch["lat"] = torch.tensor(lats, dtype=torch.float64)
            batch["lon"] = torch.tensor(lons, dtype=torch.float64)

        # Re-attach references (list of lists of strings, not tensorized)
        if references and references[0] is not None:
            batch["references"] = references

        # Re-attach image_ids (list of strings, not tensorized)
        if image_ids and image_ids[0] is not None:
            batch["image_ids"] = image_ids

        return batch
