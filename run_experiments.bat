@echo off
REM ==============================================================================
REM Physics-Residual TTA for GNN PDE Surrogates
REM 1-Click Master Benchmark Reproduction Suite
REM ==============================================================================

echo [INFO] Starting Physics-Residual TTA Reproduction Suite...
python scripts/run_all_benchmarks.py %*

echo.
echo [DONE] Reproduction finished. Check BENCHMARK_RESULTS_SUMMARY.md and benchmark_summary.csv.
pause
