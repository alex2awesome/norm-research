#!/usr/bin/env bash
# Pin + install the OFFICIAL GEPA implementation (github.com/gepa-ai/gepa, the code release of
# Agrawal et al. 2025, arXiv:2507.19457) and record exact versions for the paper.
# Everything stays inside this folder (vendor/ + a local venv) — isolation rule #2.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# official package (pip) — record the exact resolved version
pip install gepa datasets
GEPA_VER=$(python -c "import importlib.metadata as m; print(m.version('gepa'))")

# pinned source clone for provenance / reading the reference implementation
if [ ! -d vendor/gepa ]; then
  git clone https://github.com/gepa-ai/gepa vendor/gepa
fi
GEPA_SHA=$(git -C vendor/gepa rev-parse HEAD)

cat > PIN.txt <<EOF
gepa pip version: ${GEPA_VER}
gepa repo sha:    ${GEPA_SHA}
pinned on:        $(date -u +%Y-%m-%dT%H:%M:%SZ)
python:           $(python --version 2>&1)
EOF
echo "pinned:" && cat PIN.txt
