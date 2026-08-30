import types
import unittest

from scripts.download_artifacts import selected_artifacts


def _args(**overrides):
    values = {
        "all": False,
        "qwen": None,
        "satclip": False,
        "satclip_l40": False,
        "bigearthnet_encoder": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class DownloadArtifactSelectionTest(unittest.TestCase):
    def test_default_downloads_selected_smoke_artifacts(self):
        self.assertEqual(selected_artifacts(_args()), (["2B"], ["l40"], True))

    def test_all_downloads_both_satclip_variants(self):
        self.assertEqual(
            selected_artifacts(_args(all=True)),
            (["2B", "4B", "8B"], ["l10", "l40"], True),
        )

    def test_l40_can_be_selected_without_l10_or_other_artifacts(self):
        self.assertEqual(
            selected_artifacts(_args(satclip_l40=True)),
            ([], ["l40"], False),
        )


if __name__ == "__main__":
    unittest.main()
