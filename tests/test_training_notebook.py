import ast
import json
import unittest
from pathlib import Path


class TrainingNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "notebooks" / "train_and_evaluate.ipynb"
        cls.notebook = json.loads(path.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_code_cells_parse(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(source or "pass", filename=f"cell-{index}")

    def test_uses_prepared_splits_without_resplitting(self):
        self.assertIn("Upload exactly one dataset ZIP", self.code)
        self.assertIn("data/processed/arb", self.code)
        self.assertNotIn("GroupShuffleSplit", self.code)

    def test_real_dataset_training_configuration(self):
        self.assertIn("num_train_epochs=5", self.code)
        self.assertIn("max_length=512", self.code)
        self.assertIn("EarlyStoppingCallback", self.code)

    def test_reports_rewrite_and_identity_results_separately(self):
        self.assertIn("rewrite_test_predictions.csv", self.code)
        self.assertIn("identity_test_predictions.csv", self.code)
        self.assertIn("identity_review.csv", self.code)


if __name__ == "__main__":
    unittest.main()
