import unittest

from src.evaluation.bentxt_parsing import (
    bbox_iou,
    parse_bbox_answer,
    parse_binary_answer,
    parse_mcq_answer,
)


class TestBENTxTParsing(unittest.TestCase):
    def test_binary_parser_accepts_only_first_exact_answer(self) -> None:
        self.assertEqual(parse_binary_answer(" YES ").value, "yes")
        self.assertTrue(parse_binary_answer("\nno\nbecause...").extracted)

        self.assertFalse(parse_binary_answer("yes.").extracted)
        self.assertFalse(parse_binary_answer("answer: yes").extracted)

    def test_mcq_parser_accepts_only_first_exact_answer(self) -> None:
        self.assertEqual(parse_mcq_answer(" C ").value, "c")

        self.assertFalse(parse_mcq_answer("c.").extracted)
        self.assertFalse(parse_mcq_answer("option c").extracted)

    def test_bbox_parser_uses_official_format(self) -> None:
        parsed = parse_bbox_answer("[0.64 0.0, 1.0 0.71]")
        self.assertTrue(parsed.extracted)
        self.assertEqual(parsed.value, (0.64, 0.0, 1.0, 0.71))

        self.assertFalse(parse_bbox_answer("[0.64, 0.0, 1.0, 0.71]").extracted)
        self.assertFalse(parse_bbox_answer("[0.8 0.0, 0.1 0.2]").extracted)
        self.assertFalse(parse_bbox_answer("[0.0 0.0, 1.2 0.2]").extracted)

    def test_bbox_iou(self) -> None:
        iou = bbox_iou((0.0, 0.0, 1.0, 1.0), (0.5, 0.5, 1.0, 1.0))
        self.assertAlmostEqual(iou, 0.25)


if __name__ == "__main__":
    unittest.main()
