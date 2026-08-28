import unittest

from src.bentxt_grounding import (
    QWEN_OBJECT_REF_TOKENS,
    bentxt_bbox_to_qwen3_json,
    bentxt_bbox_to_qwen3_tokens,
    format_grounding_prompt,
    format_grounding_target,
    parse_qwen3_bbox,
)


class TestBENTxTGroundingConversion(unittest.TestCase):
    def test_qwen_prompt_preserves_reference_boundaries(self) -> None:
        prompt = "Identify the <ref>largest connected region of pastures</ref>."

        formatted = format_grounding_prompt(
            prompt,
            grounding_format="qwen3_json",
            ref_token=QWEN_OBJECT_REF_TOKENS,
            point_token=("", ""),
        )

        self.assertEqual(
            formatted,
            "Identify the <|object_ref_start|>largest connected region of "
            "pastures<|object_ref_end|>.",
        )

    def test_qwen_prompt_scales_point_without_flipping_axes(self) -> None:
        prompt = (
            "Output a box around the instance positioned at "
            "<point>(0.85, 0.06)</point>."
        )

        formatted = format_grounding_prompt(
            prompt,
            grounding_format="qwen3_json",
            ref_token=QWEN_OBJECT_REF_TOKENS,
            point_token=("", ""),
        )

        self.assertEqual(
            formatted,
            'Output a box around the instance positioned at {"point_2d":[850,60]}.'
        )

    def test_qwen_target_uses_bbox_json_and_1000_grid(self) -> None:
        self.assertEqual(
            bentxt_bbox_to_qwen3_json("[0.64 0.0, 1.0 0.71]"),
            '[{"bbox_2d":[640,0,1000,710]}]',
        )
        self.assertEqual(
            format_grounding_target(
                "yes",
                task_type="binary",
                grounding_format="qwen3_json",
            ),
            "yes",
        )

    def test_qwen_target_uses_pretrained_box_tokens(self) -> None:
        self.assertEqual(
            bentxt_bbox_to_qwen3_tokens("[0.64 0.0, 1.0 0.71]"),
            "<|box_start|>(640,0),(1000,710)<|box_end|>",
        )

    def test_qwen_json_round_trip_returns_bentxt_coordinates(self) -> None:
        self.assertEqual(
            parse_qwen3_bbox('{"bbox_2d":[640,0,1000,710]}'),
            (0.64, 0.0, 1.0, 0.71),
        )
        self.assertEqual(
            parse_qwen3_bbox('[{"bbox_2d":[640,0,1000,710]}]'),
            (0.64, 0.0, 1.0, 0.71),
        )

    def test_qwen_special_box_tokens_are_unwrapped_for_scoring(self) -> None:
        self.assertEqual(
            parse_qwen3_bbox(
                "<|box_start|>(640,0),(1000,710)<|box_end|>"
            ),
            (0.64, 0.0, 1.0, 0.71),
        )

    def test_rejects_invalid_or_ambiguous_qwen_boxes(self) -> None:
        self.assertIsNone(parse_qwen3_bbox('{"bbox_2d":[800,0,100,200]}'))
        self.assertIsNone(parse_qwen3_bbox("[0.64, 0.0, 1.0, 0.71]"))
        self.assertIsNone(parse_qwen3_bbox('{"bbox_2d":[640.5,0,1000,710]}'))

if __name__ == "__main__":
    unittest.main()
