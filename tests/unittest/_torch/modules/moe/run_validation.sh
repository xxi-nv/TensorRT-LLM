#!/bin/bash
# MoE Adaptive Accuracy Framework validation runner
# Usage: bash run_validation.sh <worktree_repo_path>
WT_REPO=${1:?Usage: bash run_validation.sh <worktree_repo_path>}
cd "${WT_REPO}/tests/unittest"

echo "================================================================"
echo "MoE Adaptive Accuracy Framework - GB200 Validation"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo "================================================================"

echo ""
echo "=== Phase 1: Threshold calibration ==="
python -m pytest _torch/modules/moe/test_accuracy_framework.py::test_threshold_calibration -v -s --timeout=60 2>&1 || true
P1=${PIPESTATUS[0]:-$?}

echo ""
echo "=== Phase 2: Synthetic GPU tests ==="
python -m pytest _torch/modules/moe/test_accuracy_framework.py::test_synthetic_pass -v -s --timeout=120 2>&1 || true
P2A=${PIPESTATUS[0]:-$?}
python -m pytest _torch/modules/moe/test_accuracy_framework.py::test_synthetic_fail -v -s --timeout=120 2>&1 || true
P2B=${PIPESTATUS[0]:-$?}

echo ""
echo "=== Phase 3: Full MoE integration (single GPU) ==="
python -m pytest _torch/modules/moe/test_moe_module.py -v -s -k "test_moe_module" --timeout=3600 2>&1 || true
P3=${PIPESTATUS[0]:-$?}

echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "Phase 1 (calibration):     $([ $P1 -eq 0 ] && echo PASS || echo FAIL)"
echo "Phase 2a (synthetic pass): $([ $P2A -eq 0 ] && echo PASS || echo FAIL)"
echo "Phase 2b (synthetic fail): $([ $P2B -eq 0 ] && echo PASS || echo FAIL)"
echo "Phase 3 (full MoE):        $([ $P3 -eq 0 ] && echo PASS || echo FAIL)"
echo "DONE at $(date)"
