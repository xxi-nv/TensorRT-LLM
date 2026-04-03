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

        # Prevent mpi4py from calling MPI_Init_thread (fails under enroot --mpi=none)
        export MPI4PY_RC_INITIALIZE=0
        export MPI4PY_RC_FINALIZE=0
        export TRTLLM_ENABLE_PDL=0
        export TLLM_DISABLE_MPI=1

        # Neutralize MPI/SLURM env vars that conflict
        unset PMIX_RANK PMIX_SERVER_URI PMIX_NAMESPACE PMI_RANK PMI_SIZE
        unset SLURM_MPI_TYPE
        export OMPI_MCA_plm=isolated

        echo "=== Python: $(python --version) ==="
        echo "=== Torch: $(python -c "import torch; print(torch.__version__)") ==="
        echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

        echo ""
        echo "=== Installing dependencies ==="
        pip install -q nvidia-cutlass-dsl==4.3.4 apache-tvm-ffi==0.1.6 torch-c-dlpack-ext==0.1.3 "transformers==4.57.3" 2>&1 | tail -5
        python -c "import cutlass; import cutlass.cute; print(\"cutlass import OK\")" 2>&1
        if [ $? -ne 0 ]; then
            echo "=== cutlass import FAILED ==="
            exit 1
        fi

        echo ""
        echo "=== Overlaying worktree files onto container tensorrt_llm ==="
        # Use the container pre-installed tensorrt_llm (has C++ bindings, moe_sort, etc.)
        # Copy ONLY our new/modified files on top of the container install.
        TRTLLM_PKG=$(python -c "import tensorrt_llm; print(tensorrt_llm.__path__[0])")
        echo "Container tensorrt_llm at: ${TRTLLM_PKG}"

        # New file: fused kernel
        cp -v ${WORKDIR}/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/flashmoe_fused_kernel.py \
              ${TRTLLM_PKG}/_torch/cute_dsl_kernels/blackwell/flashmoe_fused_kernel.py

        # New file: FlashMoE module
        cp -v ${WORKDIR}/tensorrt_llm/_torch/modules/fused_moe/flashmoe.py \
              ${TRTLLM_PKG}/_torch/modules/fused_moe/flashmoe.py

        # Modified file: PTX helpers used by fused kernel
        cp -v ${WORKDIR}/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py \
              ${TRTLLM_PKG}/_torch/cute_dsl_kernels/blackwell/utils.py

        # Verify overlay
        python -c "
from tensorrt_llm._torch.modules.fused_moe.flashmoe import FlashMoE
print(f\"FlashMoE imported OK from {FlashMoE.__module__}\")
"
        if [ $? -ne 0 ]; then
            echo "=== FlashMoE import FAILED after overlay ==="
            exit 1
        fi

        echo ""
        echo "=== Running correctness test (torchrun, 2 GPUs) ==="
        cd ${WORKDIR}
        torchrun --nproc_per_node=2 --master_port=29501 test_correctness.py
        RC=$?
        echo "=== Correctness test: exit code ${RC} ==="
        exit ${RC}
    '
RC=$?
echo "=== Job exit code: ${RC} ==="
