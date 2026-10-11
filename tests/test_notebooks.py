import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    PROJECT_ROOT / "notebooks" / "01_optimizer_search.ipynb",
    PROJECT_ROOT / "notebooks" / "02_transfer_baselines.ipynb",
]


class NotebookArtifactTests(unittest.TestCase):
    def test_notebooks_are_valid_and_disabled_by_default(self):
        for path in NOTEBOOKS:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertTrue(code_cells)
            for cell in code_cells:
                source = "".join(cell["source"])
                compile(source, str(path), "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                self.assertIn("RUN_EXPERIMENT = False", source)


if __name__ == "__main__":
    unittest.main()
