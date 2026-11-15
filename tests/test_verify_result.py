import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_result import verify_result


class VerifyResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "actual.bin"
        self.golden = self.root / "golden.bin"

    def check(self, tol=0.0001, expected_elements=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return verify_result(self.output, self.golden, np.float32, 0.0001, 0.0001,
                                 tol, expected_elements=expected_elements)

    def test_equal_values_pass(self):
        values = np.arange(112, dtype=np.float32)
        values.tofile(self.output)
        values.tofile(self.golden)
        self.assertTrue(self.check(expected_elements=112))

    def test_value_mismatch_fails(self):
        np.array([0.0, 1.0], dtype=np.float32).tofile(self.output)
        np.array([0.0, 2.0], dtype=np.float32).tofile(self.golden)
        self.assertFalse(self.check())

    def test_short_output_cannot_pass_under_mismatch_tolerance(self):
        values = np.arange(10001, dtype=np.float32)
        values[:-1].tofile(self.output)
        values.tofile(self.golden)
        self.assertFalse(self.check(tol=0.0001))

    def test_long_output_fails(self):
        np.arange(3, dtype=np.float32).tofile(self.output)
        np.arange(2, dtype=np.float32).tofile(self.golden)
        self.assertFalse(self.check())

    def test_partial_dtype_bytes_fail(self):
        self.output.write_bytes(b"12345")
        self.golden.write_bytes(b"12345")
        self.assertFalse(self.check())

    def test_matching_truncated_files_fail_expected_size(self):
        np.arange(111, dtype=np.float32).tofile(self.output)
        np.arange(111, dtype=np.float32).tofile(self.golden)
        self.assertFalse(self.check(expected_elements=112))

    def test_case0_cli_checks_expected_size(self):
        output_dir = self.root / "output"
        output_dir.mkdir()
        (output_dir / "y.bin").write_bytes(b"")
        (output_dir / "golden_y.bin").write_bytes(b"")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_result.py"), "0"],
                                cwd=self.root, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 112 elements", result.stdout)


if __name__ == "__main__":
    unittest.main()
