# MegaMoE Agent Evaluation Rubric

> **Purpose:** This document defines the pass/fail criteria for any agent working on the MegaMoE project. An implementation attempt is considered acceptable **if and only if** every MUST criterion below is satisfied. Any violation is an automatic fail regardless of other merits.

**Cross-references:**
- Workflow rules (memory): `feedback_megamoe_workflow.md`
- Progress tracker (memory): `project_megamoe.md`
- OCI environment (memory): `reference_oci_env.md`

---

## Hard Criteria (MUST — any miss = fail)

### C1. Kernel Technology Stack

- The MoE kernel **MUST be implemented primarily in cuteDSL** (CUTLASS 4.x Python API — `cute.jit` / `cute.kernel` / `cutlass.cute`).
- Reference implementation: DeepSeek DeepGEMM PR #304 — <https://github.com/deepseek-ai/DeepGEMM/pull/304>
- Local reference code: `/home/xxi/sc2/DeepGEMM`
- **Precision:** NVFP4 only is acceptable and expected.
- **Target platform:** NVIDIA Blackwell (sm_100+).
- **Inline PTX is allowed** where cuteDSL cannot express a needed primitive, but the body of the kernel must remain cuteDSL.
- **Fail modes:** Triton implementation, CUDA C++ implementation, falling back to existing CUTLASS C++ kernels.

### C2. Fusion Scope — "Mega" means truly fused

- MegaMoE **MUST expose a single fused Python `forward_impl`** that inlines routing, quantize, communication dispatch, grouped-GEMM compute (both L1 and L2), activation (SwiGLU), communication combine, and reduce — without going through the ConfigurableMoE 4-step method-call pipeline (`backend.quantize_input → comm.dispatch → backend.run_moe → comm.combine`).
- **Splitting communication and `run_moe` into separate ConfigurableMoE-style phases is forbidden.**
- The underlying cuteDSL kernels SHOULD progress toward fewer kernel launches (goal: merge L1 epilogue into L2 input, merge combine where feasible). Initial implementation MAY use 2–3 cuteDSL kernels; aspirational target is a single kernel per DeepGEMM PR #304.
- Inspired by DeepGEMM PR #304's fused MoE approach (communication + expert compute fused where feasible).

### C3. Class Architecture

- New code lives under a module/path clearly named **`MegaMoE`**.
- `MegaMoE` **MUST inherit directly from the MoE interface/base class** (located in `tensorrt_llm/_torch/modules/fused_moe/`).
- `MegaMoE` **MUST implement its own `forward`** — a single fused entry point.
- **`ConfigurableMoE` MUST NOT be used** as base class, helper, or composition target. Rationale: ConfigurableMoE separates communication and `run_moe`, which is structurally incompatible with the "mega fusion" goal.

### C4. Workspace Isolation — Worktree

- All modifications **MUST happen inside a dedicated git worktree** of the TRT-LLM repo.
- **Forbidden:** editing files directly in the main working tree (`/home/scratch.xxi_sw_2/trtllm` on the currently-checked-out branch).
- **Forbidden:** pushing to / overwriting the main-tree's checked-out branch.

### C5. Tests

- A new test file `tests/unittest/_torch/modules/moe/test_mega_moe.py` **MUST be added**, modeled on `test_moe_module.py`.
- Test coverage **MUST include both**:
  - Single-GPU correctness
  - Multi-GPU correctness (TP / EP where applicable)
- Reference outputs must come from an existing trusted MoE backend (numerical match within NVFP4-acceptable tolerance).

### C6. Test Execution Platform

- Tests **MUST pass on OCI B200** hardware. Local Computelab passes are insufficient.
- OCI TRTLLM worktree path: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/xxi/trtllm`
- Required workflow:
  1. Code is authored in a **Computelab** worktree.
  2. Code is pushed to a remote fork.
  3. Code is **pulled into a dedicated OCI worktree** (never into OCI's main working tree).
  4. Tests are executed on OCI B200.
- **A run is not considered "tested" until single-GPU and multi-GPU `test_mega_moe.py` both pass on OCI B200.**

### C7. Main-Repo Protection & Slurm Etiquette

- The OCI main working tree **MUST NOT be touched** — all OCI activity happens in the dedicated MegaMoE worktree.
- **Killing slurm jobs owned by the same user but started by other tasks is strictly forbidden** (`scancel` may only target jobs belonging to this MegaMoE workstream).
- Use `squeue -u $USER` for visibility only; never bulk-cancel.

### C8. Progress Persistence (Memory Discipline)

- The agent **MUST periodically record** attempted techniques, steps, decisions, and their outcomes to a memory file (`project_megamoe.md`).
- Required content for each update:
  - Necessary context for resumption (paths, branch names, current state)
  - Which techniques worked
  - Which techniques did NOT work and why
  - Current progress / next step
- **Purpose:** survive context compression, session disconnects, and agent handoffs without information loss.
- Update cadence: at minimum, after each milestone (design locked, first compile, first passing single-GPU test, first passing multi-GPU test, OCI run initiated, OCI run passed).

---

## Soft Criteria (SHOULD — counted but not fatal)

- `pre-commit run --all-files` is clean before any push.
- Code style and copyright headers follow `CODING_GUIDELINES.md`.
- PR title follows `[JIRA/NVBUG/None][type] description` convention.
- Commits are DCO-signed (`git commit -s`), without AI-tool attribution.
- Design rationale (why fused, what was split in the reference) is captured either in the source tree (a short `MEGAMOE_DESIGN.md` next to the code) or in `project_megamoe.md`.

---

## Automatic Fail Conditions

An attempt fails immediately — regardless of test results — if **any** of the following is observed:

1. Kernel written in Triton, CUDA C++, or plain PyTorch fallback rather than cuteDSL.
2. `MegaMoE` inherits from or composes with `ConfigurableMoE`, OR `forward_impl` delegates to separate `quantize_input/dispatch/run_moe/combine` method calls in the ConfigurableMoE style.
3. Target precision is not NVFP4 / target arch is not Blackwell.
4. Changes were made on the main working tree (no worktree).
5. No `test_mega_moe.py`, or tests only cover single-GPU, or tests never executed on OCI B200.
6. A slurm job belonging to another task of the same user was cancelled.
7. The OCI main working tree was modified.
8. No progress memory updates exist, or memory is silent on what was tried and what failed.

---

## Evaluation Workflow

When evaluating an attempt:

1. Verify C1–C3 by reading the MegaMoE source files.
2. Verify C4 by inspecting `git worktree list` on both Computelab and OCI.
3. Verify C5 by reading `test_mega_moe.py` and running it locally (sanity).
4. Verify C6 by inspecting slurm logs / test output on OCI B200 — both single-GPU and multi-GPU runs must show PASS.
5. Verify C7 by confirming (a) OCI main tree is unchanged, (b) no `scancel` was issued against non-MegaMoE job ids.
6. Verify C8 by reading `project_megamoe.md` — it must contain enough information for a fresh agent to resume without re-asking the user.

Any fail on C1–C8 = reject. Soft criteria failures = request revisions but may still accept.
