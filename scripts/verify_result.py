import numpy as np
import sys
import os

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

# Per-case output specs: 1 cases
# Each spec: (name, dtype, rtol, atol, tol)
# tol = max fraction of elements allowed to mismatch (0.0 = strict)
case_output_specs = {
    0: [
        ("y", np.float32, 0.0001, 0.0001, 0.0001),
    ],
}

# Case 0 output shape is (1, 7, 1, 16), as defined in main.asc.
case_expected_elements = {0: {"y": 112}}

if bfloat16 is None:
    for _cid, _specs in case_output_specs.items():
        for _name, _dtype, _rtol, _atol, _tol in _specs:
            if _dtype is None:
                print(f"ERROR: bfloat16 dtype required for case {_cid} output '{_name}' but ml_dtypes is not installed.")
                print("Install with: pip install ml_dtypes")
                sys.exit(1)

def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0, expected_elements=None):
    # Check raw byte sizes before reading: np.fromfile ignores trailing partial elements.
    output_bytes = os.path.getsize(output_path)
    golden_bytes = os.path.getsize(golden_path)
    itemsize = np.dtype(dtype).itemsize
    if output_bytes != golden_bytes:
        print(f"FAILED: output has {output_bytes} bytes, golden has {golden_bytes} — size mismatch")
        return False
    if golden_bytes % itemsize != 0:
        print(f"FAILED: file size {golden_bytes} is not a multiple of dtype size {itemsize}")
        return False
    if expected_elements is not None and golden_bytes != expected_elements * itemsize:
        print(f"FAILED: expected {expected_elements} elements, got {golden_bytes // itemsize}")
        return False

    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total_size = golden.size
    missing = 0

    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        if missing == 0 and np.array_equal(output, golden):
            print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            return True
        else:
            diff = np.abs(output.astype(np.int64) - golden.astype(np.int64))
            errors = np.sum(output != golden) + missing
            error_rate = errors / total_size if total_size > 0 else 0.0
            if error_rate <= tol:
                print(f"PASSED (tol={tol}): {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
                print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%)")
                return True
            print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
            print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
            return False

    cmp_output = output.astype(np.float32) if dtype == bfloat16 else output
    cmp_golden = golden.astype(np.float32) if dtype == bfloat16 else golden
    isclose = np.isclose(cmp_output, cmp_golden, rtol=rtol, atol=atol, equal_nan=True)
    errors = np.sum(~isclose) + missing
    error_rate = errors / total_size if total_size > 0 else 0.0
    diff = np.abs(cmp_output - cmp_golden)
    if error_rate <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return True
    else:
        print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return False

if __name__ == "__main__":
    try:
        case_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    except ValueError:
        print(f"Invalid case_id: {sys.argv[1] if len(sys.argv) > 1 else '(none)'}")
        sys.exit(1)
    if case_id not in case_output_specs:
        print(f"Unknown case_id {case_id}. Available: {sorted(case_output_specs.keys())}")
        sys.exit(1)

    output_dir = "output"
    all_pass = True
    specs = case_output_specs[case_id]
    if not specs:
        print(f"Case {case_id}: No outputs to verify (all optional) — PASS")
        sys.exit(0)
    for name, dtype, rtol, atol, tol in specs:
        output_path = os.path.join(output_dir, name + ".bin")
        golden_path = os.path.join(output_dir, "golden_" + name + ".bin")
        if not os.path.exists(output_path):
            print(f"FAILED: output {name}.bin not found")
            all_pass = False
            continue
        if not os.path.exists(golden_path):
            print(f"FAILED: golden_{name}.bin not found")
            all_pass = False
            continue
        if not verify_result(output_path, golden_path, dtype, rtol, atol, tol,
                             expected_elements=case_expected_elements.get(case_id, {}).get(name)):
            all_pass = False
    sys.exit(0 if all_pass else 1)
