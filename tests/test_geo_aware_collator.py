import unittest

import torch

from src.data_modules.geo_aware_collator import (
    DEFAULT_LOCATION_EMBED_MARKER,
    DEFAULT_LOCATION_TEXT_TEMPLATE,
    GeoAwareCollator,
)


class TestGeoAwareCollator(unittest.TestCase):
    def test_builds_chat_messages_with_system_prompt_and_re_attaches_metadata(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            system_prompt="You are a remote sensing image analysis assistant.",
        )

        items = [
            {
                "image": object(),
                "input_text": "Describe this remote sensing image.",
                "target_texts": ["caption a", "caption b"],
                "split": "bench",
                "country": "Austria",
                "season": "Spring",
                "climate_zone": "Cfb",
                "lat": 10.5,
                "lon": 20.5,
            }
        ]

        batch = collator(items)

        self.assertIn("cleaned", captured)
        messages = captured["cleaned"][0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"][0]["text"], "caption a")

        self.assertIn("lat", batch)
        self.assertIn("lon", batch)
        self.assertIn("target_texts", batch)
        self.assertEqual(batch["target_texts"], [["caption a", "caption b"]])
        self.assertEqual(batch["input_text"], ["Describe this remote sensing image."])
        self.assertEqual(batch["split"], ["bench"])
        self.assertEqual(batch["country"], ["Austria"])
        self.assertEqual(batch["season"], ["Spring"])
        self.assertEqual(batch["climate_zone"], ["Cfb"])
        self.assertTrue(torch.equal(batch["lat"], torch.tensor([10.5], dtype=torch.float64)))
        self.assertTrue(torch.equal(batch["lon"], torch.tensor([20.5], dtype=torch.float64)))

    def test_appends_location_text_without_changing_native_prompt_source(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            location_text_template="Capture location: latitude {lat:.4f}, longitude {lon:.4f}.",
        )

        collator(
            [
                {
                    "image": object(),
                    "input_text": "Which land cover classes are present?",
                    "target_texts": ["Urban fabric"],
                    "lat": 52.12346,
                    "lon": 13.98765,
                }
            ]
        )

        user_text = captured["cleaned"][0]["messages"][0]["content"][0]["text"]
        self.assertEqual(
            user_text,
            "Which land cover classes are present?\n"
            "Capture location: latitude 52.1235, longitude 13.9877.",
        )

    def test_appends_compact_integer_location_text(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            location_text_template=DEFAULT_LOCATION_TEXT_TEMPLATE,
        )

        collator(
            [
                {
                    "image": object(),
                    "input_text": "Which land cover classes are present?",
                    "target_texts": ["Urban fabric"],
                    "lat": -52.12346,
                    "lon": 13.98765,
                }
            ]
        )

        user_text = captured["cleaned"][0]["messages"][0]["content"][0]["text"]
        self.assertEqual(
            user_text,
            "Which land cover classes are present?\n"
            "Scene coordinates: 52°S, 14°E.",
        )

    def test_appends_compact_decimal_location_text(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            location_text_template=DEFAULT_LOCATION_TEXT_TEMPLATE,
            coordinates_decimal_places=2,
        )

        collator(
            [
                {
                    "image": object(),
                    "input_text": "Which land cover classes are present?",
                    "target_texts": ["Urban fabric"],
                    "lat": -52.12346,
                    "lon": 13.98765,
                }
            ]
        )

        user_text = captured["cleaned"][0]["messages"][0]["content"][0]["text"]
        self.assertEqual(
            user_text,
            "Which land cover classes are present?\n"
            "Scene coordinates: 52.12°S, 13.99°E.",
        )

    def test_rejects_negative_coordinates_decimal_places(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinates_decimal_places"):
            GeoAwareCollator(
                lambda cleaned: {},
                location_text_template=DEFAULT_LOCATION_TEXT_TEMPLATE,
                coordinates_decimal_places=-1,
            )

    def test_appends_location_embed_marker_without_coordinate_text(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            location_text_template=DEFAULT_LOCATION_EMBED_MARKER,
        )

        collator(
            [
                {
                    "image": object(),
                    "input_text": "Which land cover classes are present?",
                    "target_texts": ["Urban fabric"],
                    "lat": -52.12346,
                    "lon": 13.98765,
                }
            ]
        )

        user_text = captured["cleaned"][0]["messages"][0]["content"][0]["text"]
        self.assertEqual(
            user_text,
            "Which land cover classes are present?\n"
            "Scene coordinates:",
        )

    def test_re_attaches_non_rgb_tensor_without_sending_it_to_unsloth(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}

        collator = GeoAwareCollator(inner_collator)
        first_non_rgb = torch.ones(3, 2, 2)
        second_non_rgb = torch.zeros(3, 2, 2)

        batch = collator(
            [
                {
                    "image": object(),
                    "input_text": "Describe image A.",
                    "target_texts": ["caption a"],
                    "lat": 10.5,
                    "lon": 20.5,
                    "non_rgb_imagery": first_non_rgb,
                    "non_rgb_bands": ["VV", "VH", "B04", "B03", "B02"],
                },
                {
                    "image": object(),
                    "input_text": "Describe image B.",
                    "target_texts": ["caption b"],
                    "lat": -1.0,
                    "lon": 42.0,
                    "non_rgb_imagery": second_non_rgb,
                    "non_rgb_bands": ["VV", "VH", "B04", "B03", "B02"],
                },
            ]
        )

        self.assertNotIn("non_rgb_imagery", captured["cleaned"][0])
        self.assertTrue(torch.equal(
            batch["non_rgb_imagery"],
            torch.stack([first_non_rgb, second_non_rgb], dim=0),
        ))
        self.assertEqual(batch["non_rgb_bands"], ["VV", "VH", "B04", "B03", "B02"])

    def test_rejects_partial_non_rgb_batches(self) -> None:
        collator = GeoAwareCollator(lambda cleaned: {})

        with self.assertRaisesRegex(ValueError, "Either every sample or no sample"):
            collator(
                [
                    {
                        "image": object(),
                        "input_text": "Describe image A.",
                        "target_texts": ["caption a"],
                        "lat": 10.5,
                        "lon": 20.5,
                        "non_rgb_imagery": torch.ones(3, 2, 2),
                    },
                    {
                        "image": object(),
                        "input_text": "Describe image B.",
                        "target_texts": ["caption b"],
                        "lat": -1.0,
                        "lon": 42.0,
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
