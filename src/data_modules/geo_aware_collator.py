import math
from typing import Any

import torch


DEFAULT_LOCATION_TEXT_TEMPLATE = "Scene coordinates: {location}."
DEFAULT_LOCATION_EMBED_MARKER = "Scene coordinates:"


def _location_template_fields(lat: float, lon: float) -> dict[str, Any]:
    return {
        "lat": lat,
        "lon": lon,
        "location": _compact_integer_location(lat, lon),
    }


def _rounded_abs_degrees(value: float) -> int:
    return math.floor(abs(value) + 0.5)


def _compact_integer_location(lat: float, lon: float) -> str:
    lat_hemisphere = "N" if lat >= 0 else "S"
    lon_hemisphere = "E" if lon >= 0 else "W"
    return (
        f"{_rounded_abs_degrees(lat)}°{lat_hemisphere}, "
        f"{_rounded_abs_degrees(lon)}°{lon_hemisphere}"
    )


class GeoAwareCollator:
    """Wraps the Unsloth vision collator for normalized samples.

    Input sample schema:
      - image
      - input_text
      - target_texts
      - lat
      - lon
      - sample_id (optional)
      - patch_id (optional)
      - task_type (optional)
      - task_category (optional)
      - non_rgb_imagery (optional)
      - non_rgb_bands (optional)

    The inner Unsloth collator expects chat-style `messages`, so this wrapper
    constructs those messages and re-attaches metadata needed later in the
    model/evaluation path.
    """

    def __init__(
        self,
        inner_collator,
        system_prompt: str | None = None,
        location_text_template: str | None = None,
    ):
        self.inner_collator = inner_collator
        self.system_prompt = system_prompt
        self.location_text_template = location_text_template

    def _to_messages(self, image: Any, input_text: str, target_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ])
        return messages

    def _with_location_text(self, input_text: str, lat: float, lon: float) -> str:
        if not self.location_text_template:
            return input_text
        location_text = self.location_text_template.format(**_location_template_fields(lat, lon))
        return f"{input_text}\n{location_text}"

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert normalized samples to the message format expected by Unsloth.
        lats, lons, target_texts_batch = [], [], []
        optional_metadata = {
            "sample_id": [],
            "patch_id": [],
            "task_type": [],
            "task_category": [],
        }
        non_rgb_images = []
        non_rgb_bands = []
        has_non_rgb_imagery = ["non_rgb_imagery" in item for item in items]
        if any(has_non_rgb_imagery) and not all(has_non_rgb_imagery):
            raise ValueError("Either every sample or no sample must include 'non_rgb_imagery'")

        cleaned = []
        for item in items:
            image = item["image"]
            input_text = str(item["input_text"])
            targets = [str(t) for t in item["target_texts"]]
            if not targets:
                raise ValueError("Each sample must contain at least one target text")

            lat = item["lat"]
            lon = item["lon"]
            if not isinstance(lat, float):
                raise TypeError(
                    "Expected 'lat' to be float in normalized sample, "
                    f"got {type(lat).__name__}: {lat!r}"
                )
            if not isinstance(lon, float):
                raise TypeError(
                    "Expected 'lon' to be float in normalized sample, "
                    f"got {type(lon).__name__}: {lon!r}"
                )
            lats.append(lat)
            lons.append(lon)
            target_texts_batch.append(targets)
            for key, values in optional_metadata.items():
                if key in item:
                    values.append(item[key])
            input_text = self._with_location_text(input_text, lat, lon)
            if has_non_rgb_imagery[0]:
                non_rgb_image = item["non_rgb_imagery"]
                if not isinstance(non_rgb_image, torch.Tensor):
                    raise TypeError(
                        "Expected 'non_rgb_imagery' to be a torch.Tensor in normalized sample, "
                        f"got {type(non_rgb_image).__name__}"
                    )
                non_rgb_images.append(non_rgb_image)
                non_rgb_bands.append(item.get("non_rgb_bands"))

            cleaned.append({"messages": self._to_messages(image, input_text, targets[0])})

        # Inner collation (input_ids, attention_mask, labels, pixel_values, ...)
        batch = self.inner_collator(cleaned)

        # Re-attach geo tensors and sample metadata for conditioning/evaluation.
        batch["lat"] = torch.tensor(lats, dtype=torch.float64)
        batch["lon"] = torch.tensor(lons, dtype=torch.float64)

        # Keep full targets for multi-reference evaluation.
        batch["target_texts"] = target_texts_batch
        for key, values in optional_metadata.items():
            if values:
                if len(values) != len(items):
                    raise ValueError(f"Either every sample or no sample must include {key!r}")
                batch[key] = [str(value) for value in values]
        if non_rgb_images:
            batch["non_rgb_imagery"] = torch.stack(non_rgb_images, dim=0)
            if any(bands is not None for bands in non_rgb_bands):
                if all(bands == non_rgb_bands[0] for bands in non_rgb_bands):
                    batch["non_rgb_bands"] = non_rgb_bands[0]
                else:
                    batch["non_rgb_bands"] = non_rgb_bands

        return batch
