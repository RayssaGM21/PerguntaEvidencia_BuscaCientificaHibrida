from __future__ import annotations

import unittest

from src.config import BEIR_DATASETS


class LoaderConfigTest(unittest.TestCase):
    def test_required_datasets_are_configured(self) -> None:
        self.assertIn("nfcorpus", BEIR_DATASETS)
        self.assertIn("scifact", BEIR_DATASETS)
        self.assertIn("trec-covid", BEIR_DATASETS)


if __name__ == "__main__":
    unittest.main()
