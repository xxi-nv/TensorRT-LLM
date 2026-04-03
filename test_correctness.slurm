#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=xxi-correct
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe/correct_%j.log
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm-wt-flashmoe/correct_%j.log

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
        echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

        echo ""
        echo "=== Installing dependencies ==="
        pip install -q nvidia-cutlass-dsl==4.3.4 apache-tvm-ffi==0.1.6 torch-c-dlpack-ext==0.1.3 2>&1 | tail -5
        python -c "import cutlass; import cutlass.cute; print(\"cutlass import OK\")" 2>&1
        if [ $? -ne 0 ]; then
            echo "=== cutlass import FAILED ==="
            exit 1
        fi

        echo ""
        echo "=== Running single-GPU correctness test ==="
        python test_correctness.py
        RC=$?
        echo "=== Correctness test: exit code ${RC} ==="
        exit ${RC}
    '
RC=$?
echo "=== Job exit code: ${RC} ==="
