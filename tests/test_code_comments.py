import tokenize
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CodeCommentTests(unittest.TestCase):
    def test_python_files_contain_no_comment_tokens(self):
        python_files = list(PROJECT_ROOT.rglob("*.py"))
        for path in python_files:
            with path.open("r", encoding="utf-8") as stream:
                tokens = tokenize.generate_tokens(stream.readline)
                comments = [token for token in tokens if token.type == tokenize.COMMENT]
            self.assertEqual(comments, [], str(path.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    unittest.main()
