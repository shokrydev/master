from typing import Any

import torch


class GeoAwareCollator:
    """Wraps an inner collator (e.g. UnslothVisionDataCollator) to handle geo fields.

    Before inner collation: strips lat, lon, image_id from items (the inner
    collator only understands 'messages').
    After inner collation: re-attaches GAIA metadata needed later in the
    pipeline. `references` and `image_ids` are preserved for evaluation in all
    conditions. `lat` / `lon` are only attached when `include_coordinates=True`,
    which is the path used by location-conditioned runs including SatCLIP
    `loc_embed`.
    """

    def __init__(self, inner_collator, include_coordinates: bool = False):
        self.inner_collator = inner_collator
        self.include_coordinates = include_coordinates

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        # Strip extra fields before passing to inner collator
        lats, lons, image_ids, references = [], [], [], []
        cleaned = []
        for item in items:
            item = dict(item)  # shallow copy
            image_ids.append(item.pop("image_id", None))
            references.append(item.pop("references", None))
            if self.include_coordinates:
                lats.append(item.pop("lat", None))
                lons.append(item.pop("lon", None))
            else:
                item.pop("lat", None)
                item.pop("lon", None)
            cleaned.append(item)

        # Inner collation (produces input_ids, attention_mask, labels, pixel_values, etc.)
        batch = self.inner_collator(cleaned)

        # Re-attach geo tensors
        if self.include_coordinates and lats and lats[0] is not None:
            batch["lat"] = torch.tensor(lats, dtype=torch.float64)
            batch["lon"] = torch.tensor(lons, dtype=torch.float64)

        # Re-attach references (list of lists of strings, not tensorized)
        if references and references[0] is not None:
            batch["references"] = references

        # Re-attach image_ids (list of strings, not tensorized)
        if image_ids and image_ids[0] is not None:
            batch["image_ids"] = image_ids

        return batch
