# ShiftBench harness

Evaluation code for [docs/benchmark/SHIFTBENCH.md](../../docs/benchmark/SHIFTBENCH.md).
Python (evaluation only; TRACE is called through the `traceot` bindings).

    pip install numpy scipy pandas scikit-learn lightgbm
    cd benchmarks/shiftbench
    python run_synthetic.py truth
    python run_synthetic.py accuracy
    python run_synthetic.py coverage
    python run_synthetic.py invariance
    python run_synthetic.py scale
    python run_real.py
    python summarize.py

Files: `sb_dgp.py` (scenarios and oracle truth), `sb_methods.py` (rivals),
`sb_adapter.py` (method interface and TRACE slot), `data/` (derived real panel).
