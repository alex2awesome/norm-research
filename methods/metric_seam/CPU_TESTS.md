# Metric-seam CPU verification

Run the canonical CPU-only suite from the repository root:

```bash
python -m methods.metric_seam.run_cpu_tests
```

The runner uses the current Python interpreter, fixes the working directory at the
repository root, removes API credentials from child environments, and masks CUDA, NVIDIA,
HIP, and ROCm devices. It runs the metric-seam pytest suite, the historical certificate
battery, the standalone 43-check capability-v1 battery, and the complete capability-v2.1
counterexample replay. It does not make model/API calls or rewrite scientific artifacts.

To inspect the exact commands without running them:

```bash
python -m methods.metric_seam.run_cpu_tests --dry-run
```

The frozen Math-a12 v1 replay integration is marked non-strict `xfail`: its historical
0.5-second wall-clock parser budget can be exceeded by cold SymPy/Lark initialization on a
loaded CPU. Its assertions and frozen source remain unchanged; a fast run appears as `XPASS`.
