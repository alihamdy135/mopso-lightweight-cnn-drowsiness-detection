import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_release_tree_has_no_docs_directory(self):
        self.assertFalse((PROJECT_ROOT / "docs").exists())

    def test_expected_project_artifacts_exist(self):
        required = [
            "README.md",
            "CITATION.cff",
            "configs/paper_default.yaml",
            "src/mopso_drowsiness/search/algorithms/ga.py",
            "src/mopso_drowsiness/search/algorithms/pso.py",
            "src/mopso_drowsiness/search/algorithms/mopso.py",
            "src/mopso_drowsiness/search/algorithms/nsga2.py",
            "notebooks/01_optimizer_search.ipynb",
            "notebooks/02_transfer_baselines.ipynb",
        ]
        self.assertTrue(all((PROJECT_ROOT / item).is_file() for item in required))


if __name__ == "__main__":
    unittest.main()
