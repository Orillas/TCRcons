#!/bin/bash
set -e
PY=/home/jilin/DeepTCR/.venv/bin/python3
SDIR=/home/jilin/DeepTCR/tcrconsensus/scripts
LOGDIR=/home/jilin/DeepTCR/tcrconsensus/results/p0_experiments

echo "=== $(date) Starting P0 experiments (7 methods) ==="

echo "=== $(date) [1/4] Case Study (GILGFVFTL + ELAGIGILTV) ==="
mkdir -p $LOGDIR/case_study_7m
cd $SDIR
$PY -u p0_case_study.py > $LOGDIR/case_study_7m/case_study_7m.log 2>&1
echo "=== $(date) [1/4] Case Study DONE ==="

echo "=== $(date) [2/4] Algorithm Ablation ==="
mkdir -p $LOGDIR/algorithm_ablation_7m
cd $SDIR
$PY -u p0_algorithm_ablation.py > $LOGDIR/algorithm_ablation_7m/ablation_7m.log 2>&1
echo "=== $(date) [2/4] Algorithm Ablation DONE ==="

echo "=== $(date) [3/4] Improved Stress Test (6 subsets) ==="
mkdir -p $LOGDIR/improved_stress_7m
cd $SDIR
$PY -u p0_improved_stress.py > $LOGDIR/improved_stress_7m/stress_7m.log 2>&1
echo "=== $(date) [3/4] Improved Stress Test DONE ==="

echo "=== $(date) [4/4] CV Weight Learning (6 subsets) ==="
mkdir -p $LOGDIR/cv_weights_7m
cd $SDIR
$PY -u p0_cv_weights.py > $LOGDIR/cv_weights_7m/cv_weights_7m.log 2>&1
echo "=== $(date) [4/4] CV Weight Learning DONE ==="

echo "=== $(date) ALL P0 EXPERIMENTS COMPLETE ==="
