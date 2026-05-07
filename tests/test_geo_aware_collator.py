import unittest

import torch

from src.data_modules.geo_aware_collator import GeoAwareCollator


class TestGeoAwareCollator(unittest.TestCase):
    def test_builds_chat_messages_with_system_prompt_and_re_attaches_metadata(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        collator = GeoAwareCollator(
            inner_collator,
            include_coordinates=True,
            system_prompt="You are a remote sensing image analysis assistant.",
        )

        items = [
            {
                "image": object(),
                "input_text": "Describe this remote sensing image.",
                "target_texts": ["caption a", "caption b"],
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
        self.assertTrue(torch.equal(batch["lat"], torch.tensor([10.5], dtype=torch.float64)))
        self.assertTrue(torch.equal(batch["lon"], torch.tensor([20.5], dtype=torch.float64)))

    def test_re_attaches_multispectral_tensor_without_sending_it_to_unsloth(self) -> None:
        captured = {}

        def inner_collator(cleaned):
            captured["cleaned"] = cleaned
            return {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}

        collator = GeoAwareCollator(inner_collator, include_coordinates=False)
        first_multispectral = torch.ones(3, 2, 2)
        second_multispectral = torch.zeros(3, 2, 2)

        batch = collator(
            [
                {
                    "image": object(),
                    "input_text": "Describe image A.",
                    "target_texts": ["caption a"],
                    "lat": 10.5,
                    "lon": 20.5,
                    "multispectral": first_multispectral,
                    "multispectral_bands": ["B04", "B03", "B02"],
                },
                {
                    "image": object(),
                    "input_text": "Describe image B.",
                    "target_texts": ["caption b"],
                    "lat": -1.0,
                    "lon": 42.0,
                    "multispectral": second_multispectral,
                    "multispectral_bands": ["B04", "B03", "B02"],
                },
            ]
        )

        self.assertNotIn("multispectral", captured["cleaned"][0])
        self.assertTrue(torch.equal(
            batch["multispectral"],
            torch.stack([first_multispectral, second_multispectral], dim=0),
        ))
        self.assertEqual(batch["multispectral_bands"], ["B04", "B03", "B02"])

    def test_rejects_partial_multispectral_batches(self) -> None:
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
                        "multispectral": torch.ones(3, 2, 2),
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
