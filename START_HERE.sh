#!/usr/bin/env bash
# One command does everything for the final comparison. It makes its own Python environment inside this folder (first time
# only), checks the code, runs the new 5-split pipeline on the same 240 data sets as your final run, runs it on the 10 CDC
# samples, and writes results/FINAL.md. About 1 hour on 2 cores. Nothing outside this folder is touched.
#
#   cd into this folder, then:      caffeinate -i bash START_HERE.sh
#
# Stopped halfway? Run the same command again: finished settings are skipped.
set -e
cd "$(dirname "$0")"

if [ -z "$SKIP_INSTALL" ]; then
  echo "== 1/5  Python environment (first time only)"
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt threadpoolctl || pip install --quiet numpy pandas scikit-learn scipy threadpoolctl pytest
fi

echo "== 2/5  Checking the code (68 tests, about 2 minutes)"
python -m pytest -q tests

echo "== 3/5  The 5-split pipeline on the 240 data sets of the final run (about 40 to 60 minutes; one line per finished setting)"
python -m select_then_match.final sim --reps "${REPS:-20}"

echo "== 4/5  The 10 CDC samples (about 5 to 10 minutes)"
python -m select_then_match.final cdc --reps "${CDC_REPS:-10}"

echo "== 5/5  Report"
python -m select_then_match.final report
echo
echo "DONE. Send me this file:  $(pwd)/results/FINAL.md"
