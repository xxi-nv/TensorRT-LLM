#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=xxi-mgpu-ar
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe/mgpu_ar_%j.log
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe/mgpu_ar_%j.log

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/common/trtllm_sqsh/trtllm-pytorch-25.12-py3-aarch64-ubuntu24.04-trt10.14.1.48-skip-tritondevel-202603051044-11898.sqsh

echo "=== Job ${SLURM_JOBID} on $(hostname) ==="
echo "=== Commit: $(cd ${WORKDIR} && git log --oneline -1) ==="

srun --mpi=none --ntasks=1 --ntasks-per-node=1 \
    --container-image="${CONTAINER}" \
    --container-mounts="${WORKDIR}:${WORKDIR}" \
    bash -c '
        WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe
        export PYTHONPATH=${WORKDIR}:${PYTHONPATH}
        export TRTLLM_ENABLE_PDL=0
        export TLLM_DISABLE_MPI=1
        cd ${WORKDIR}

        echo "=== Python: $(python --version) ==="
        echo "=== Torch: $(python -c "import torch; print(torch.__version__)") ==="
        echo "=== GPUs: ==="
        nvidia-smi --query-gpu=index,name --format=csv,noheader

        echo ""
        echo "=== Installing dependencies ==="
        pip install -q nvidia-cutlass-dsl==4.3.4 apache-tvm-ffi==0.1.6 torch-c-dlpack-ext==0.1.3 2>&1 | tail -5
        python -c "import cutlass; import cutlass.cute; print(\"cutlass import OK\")" 2>&1
        if [ $? -ne 0 ]; then
            echo "=== cutlass import FAILED ==="
            exit 1
        fi

        echo ""
        echo "=== P2P accessibility check ==="
        python -c "
import torch
n = torch.cuda.device_count()
print(f\"GPUs available: {n}\")
for i in range(n):
    for j in range(n):
        if i != j:
            can = torch.cuda.can_device_access_peer(i, j)
            print(f\"  GPU {i} -> GPU {j}: P2P={can}\")
"

        echo ""
        echo "=== Test 1: 2-GPU cross-device AR ==="
        python test_multi_gpu_ar.py --num-gpus 2
        RC2=$?
        echo "=== 2-GPU test: exit code ${RC2} ==="

        echo ""
        echo "=== Test 2: 4-GPU cross-device AR ==="
        python test_multi_gpu_ar.py --num-gpus 4
        RC4=$?
        echo "=== 4-GPU test: exit code ${RC4} ==="

        echo ""
        echo "=== Test 3: 2-process IPC AR (real multi-process) ==="
        python test_multiproc_ar.py --num-gpus 2
        RC_MP2=$?
        echo "=== 2-proc IPC test: exit code ${RC_MP2} ==="

        if [ ${RC2} -ne 0 ] || [ ${RC4} -ne 0 ] || [ ${RC_MP2} -ne 0 ]; then
            echo "=== SOME TESTS FAILED ==="
            exit 1
        fi
        echo "=== ALL MULTI-GPU AR TESTS PASSED ==="
        exit 0
    '
RC=$?
echo "=== Job exit code: ${RC} ==="
