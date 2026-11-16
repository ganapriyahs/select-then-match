#!/usr/bin/env bash
# Everything, in order. Run from this folder:  bash run_all.sh
# Needs the CDC csv in data/ (see data/README.md). About 40 minutes on 2 cores (the simulation is most of it).
set -e
python -m select_then_match.step1                                   # Step 1: descriptive statistics
python -m select_then_match.step2                                   # Step 2: regression and variable screening
python -m select_then_match.step3 sim --reps 20 --seed0 6000        # Step 3: ten methods, 12 simulated settings
python -m select_then_match.step3 cdc --n 4000 --reps 10            # Step 3: the same ten methods on CDC samples
python -m select_then_match.step3 report                            # results/RESULTS.md
