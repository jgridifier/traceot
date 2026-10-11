"""Architecture rule: all numerical methods live in Rust; Python is a thin binding.

These checks are a guard rail, not a proof: they forbid the numpy calls that would
signal numerical logic creeping into the runtime package, and check that the test-only
reference is never imported by it.
"""

import ast
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "python" / "traceot"

# numpy functions that do arithmetic/statistics/linear algebra (validation helpers such
# as asarray, isfinite, any/all, broadcast_to, full or ascontiguousarray are allowed).
FORBIDDEN = {
    "mean",
    "sum",
    "dot",
    "matmul",
    "einsum",
    "linalg",
    "interp",
    "sort",
    "argsort",
    "quantile",
    "percentile",
    "median",
    "exp",
    "log",
    "sqrt",
    "cumsum",
    "diff",
    "ptp",
    "argmin",
    "argmax",
    "searchsorted",
    "histogram",
    "bincount",
    "random",
    "geomspace",
    "linspace",
    "setdiff1d",
    "var",
    "std",
    "cov",
    "polyfit",
    "convolve",
    "fft",
}


def _py_files():
    return sorted(PKG.rglob("*.py"))


def test_no_numerical_numpy_calls_in_package():
    bad = []
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                bad.append(f"{f.name}:{node.lineno} uses @")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"np", "numpy"}
                and node.attr in FORBIDDEN
            ):
                bad.append(f"{f.name}:{node.lineno} np.{node.attr}")
    assert not bad, "numerical logic belongs in crates/core: " + ", ".join(bad)


def test_package_never_imports_reference():
    for f in _py_files():
        assert "reference" not in {
            (n.module or "").split(".")[0]
            for n in ast.walk(ast.parse(f.read_text()))
            if isinstance(n, ast.ImportFrom)
        }, f
    code = "import sys, traceot; print(any(m.split('.')[0] == 'reference' for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
