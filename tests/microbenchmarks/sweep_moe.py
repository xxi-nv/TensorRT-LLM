#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Outer sweep driver for ``bench_moe.py``.

Splits a multi-axis MoE sweep into one ``bench_moe.py`` invocation per
``(parallel_mode, num_tokens)`` leaf so that a CUDA context pollution (e.g.
``cudaErrorIllegalInstruction``) inside one leaf cannot contaminate the
others. Each leaf runs under its own ``mpirun``/``srun`` (own CUDA context),
checkpoints incrementally via ``bench_moe.py --output_file``, and is
auto-resumed on transient failures:

  * Exit code 75: ``bench_moe.py`` detected a sticky CUDA error after a
    candidate and voluntarily exited so the outer driver could restart.
  * Exit code 137 / -9 / -SIGKILL: ``bench_moe.py``'s per-candidate watchdog
    (``--per_candidate_timeout_s``) tripped on a suspected NCCL deadlock.

Both are treated as retryable. The driver re-runs the leaf with
``--resume_from <leaf>/result.json`` so already-completed candidates are
skipped. After ``--per_leaf_max_retries`` consecutive retryable failures the
leaf is marked permanently failed and the next leaf starts.

Example:
  python tests/microbenchmarks/sweep_moe.py \\
      --launcher "mpirun -np 4" \\
      --world_size 4 \\
      --output_dir oci_runs/moe_sweep \\
      --parallel_modes DEP TEP DTP TTP \\
      --num_tokens 4 64 512 1024 4096 \\
      --per_leaf_max_retries 2 \\
      -- \\
      --model deepseek_v4_pro --search backend comm --backend ALL \\
      --per_candidate_timeout_s 180 --warmup 1 --iters 5

Everything after the ``--`` is passed verbatim to ``bench_moe.py``; the
driver only overrides ``--parallel_mode``, ``--num_tokens``, ``--world_size``,
``--output_file``, and ``--resume_from`` per leaf.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Exit-code conventions matched against ``bench_moe.py``:
#   75: ``_BENCH_MOE_POISON_EXIT_CODE`` -- voluntary exit on sticky CUDA error.
#   137 / -9 / -SIGKILL: process killed by ``_CandidateWatchdog``.
# Any of these indicates a transient pollution / hang that a fresh CUDA
# context + resume should recover from.
_BENCH_MOE_POISON_EXIT = 75
_RETRYABLE_EXIT_CODES = {
    _BENCH_MOE_POISON_EXIT,
    137,
    -9,
    -int(signal.SIGKILL),
}

_DEFAULT_BENCH_PATH = Path(__file__).resolve().parent / "bench_moe" / "__main__.py"

# Args we always set per-leaf; user-supplied copies of these are stripped from
# the pass-through bench_args to avoid silent argparse override surprises.
_DRIVER_OWNED_BENCH_ARGS = frozenset(
    {
        "--parallel_mode",
        "--num_tokens",
        "--world_size",
        "--output_file",
        "-o",
        "--resume_from",
    }
)


def _parse_driver_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Outer sweep driver for bench_moe.py. Splits a sweep into one "
            "mpirun-per-(parallel_mode, num_tokens) leaf so CUDA context "
            "pollution in one leaf cannot poison others; auto-resumes on "
            "transient sticky-error (exit 75) or watchdog SIGKILL exits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--launcher",
        required=True,
        help=(
            "Shell-quoted MPI launcher prefix, e.g. 'mpirun -np 4' or "
            "'srun --ntasks=4 --gpus-per-node=4'. Prepended verbatim to every "
            "bench_moe invocation."
        ),
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="World size; must match the launcher's task count. Forwarded to bench_moe.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Per-sweep output dir. Each leaf gets its own subdir; combined results land here.",
    )
    parser.add_argument(
        "--parallel_modes",
        nargs="+",
        required=True,
        choices=("DEP", "TEP", "DTP", "TTP", "CUSTOM"),
        help="Parallel modes to sweep, one mpirun per mode.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        nargs="+",
        required=True,
        help="num_tokens values to sweep, one mpirun per value per parallel_mode.",
    )
    parser.add_argument(
        "--bench_script",
        default=str(_DEFAULT_BENCH_PATH),
        help=(
            "Path to the bench_moe entry script. Defaults to "
            "bench_moe/__main__.py next to this driver."
        ),
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help=(
            "Python interpreter to invoke under the launcher. Defaults to the "
            "interpreter running this driver. Override when the driver runs on "
            "a login node but the mpirun launches into a container with a "
            "different Python (e.g. '/usr/bin/python3' inside the TRT-LLM container)."
        ),
    )
    parser.add_argument(
        "--per_leaf_max_retries",
        type=int,
        default=2,
        help=(
            "Max retry attempts per leaf when bench_moe exits with a transient "
            "failure code (sticky CUDA error or watchdog SIGKILL). Each retry "
            "reuses the leaf's checkpoint JSON via --resume_from."
        ),
    )
    parser.add_argument(
        "--per_leaf_retry_backoff_s",
        type=float,
        default=15.0,
        help="Sleep this long between retries to let the GPU / NCCL state settle.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print every command that would be run, but don't execute them.",
    )
    parser.add_argument(
        "--no_combined_report",
        action="store_true",
        help="Skip the post-sweep merge into combined_result.json.",
    )
    parser.add_argument(
        "--continue_on_leaf_failure",
        action="store_true",
        default=True,
        help=(
            "Move on to the next leaf when a leaf exhausts its retries. "
            "Default True so one bad parallel_mode does not abort the whole sweep."
        ),
    )
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help=(
            "Pass-through args for bench_moe.py. Use '--' to separate, e.g. "
            "sweep_moe.py ... -- --model deepseek_v4_pro --search backend comm "
            "--backend ALL --per_candidate_timeout_s 180."
        ),
    )
    return parser.parse_args(argv)


def _leaf_name(parallel_mode: str, num_tokens: int) -> str:
    return f"{parallel_mode}_nt{num_tokens}"


def _leaf_dir(output_dir: Path, parallel_mode: str, num_tokens: int) -> Path:
    return output_dir / _leaf_name(parallel_mode, num_tokens)


def _strip_driver_owned_args(bench_args: List[str]) -> List[str]:
    """Drop driver-owned flag-and-value pairs from the pass-through args.

    The driver always overrides ``--parallel_mode``, ``--num_tokens``,
    ``--world_size``, ``--output_file``, and ``--resume_from`` per leaf.
    Silently dropping user-supplied copies avoids argparse precedence surprises
    where ``bench_moe.py`` would accept the user's value and ignore ours.
    """
    out: List[str] = []
    i = 0
    while i < len(bench_args):
        tok = bench_args[i]
        if tok == "--":
            i += 1
            continue
        if tok in _DRIVER_OWNED_BENCH_ARGS:
            i += 1
            # Consume the value(s) until the next --flag.
            while i < len(bench_args) and not bench_args[i].startswith("-"):
                i += 1
            continue
        if "=" in tok and tok.split("=", 1)[0] in _DRIVER_OWNED_BENCH_ARGS:
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _build_leaf_cmd(
    *,
    args: argparse.Namespace,
    parallel_mode: str,
    num_tokens: int,
    leaf_dir: Path,
) -> List[str]:
    bench_extra = _strip_driver_owned_args(list(args.bench_args or []))
    leaf_json = leaf_dir / "result.json"
    return (
        shlex.split(args.launcher)
        + [
            args.python_executable,
            args.bench_script,
            "--parallel_mode",
            parallel_mode,
            "--num_tokens",
            str(num_tokens),
            "--world_size",
            str(args.world_size),
            "--output_file",
            str(leaf_json),
            "--resume_from",
            str(leaf_json),
        ]
        + bench_extra
    )


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _is_retryable_rc(rc: int) -> bool:
    return rc in _RETRYABLE_EXIT_CODES


def _run_one_leaf(
    *,
    args: argparse.Namespace,
    parallel_mode: str,
    num_tokens: int,
    leaf_dir: Path,
    driver_log,
) -> Dict[str, Any]:
    """Run one leaf with retry-on-transient-failure semantics.

    Returns a stats dict recording every attempt and the final exit code.
    """
    leaf_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_leaf_cmd(
        args=args,
        parallel_mode=parallel_mode,
        num_tokens=num_tokens,
        leaf_dir=leaf_dir,
    )
    stdout_log = leaf_dir / "stdout.log"
    stderr_log = leaf_dir / "stderr.log"
    stats_path = leaf_dir / "run_stats.json"

    attempts: List[Dict[str, Any]] = []
    final_rc: Optional[int] = None
    label = f"{parallel_mode} nt={num_tokens}"
    max_attempts = max(1, int(args.per_leaf_max_retries) + 1)

    for attempt_idx in range(1, max_attempts + 1):
        start = time.monotonic()
        msg = f"[sweep_moe] {label}: attempt {attempt_idx}/{max_attempts}: {_format_cmd(cmd)}"
        print(msg, flush=True)
        print(msg, file=driver_log, flush=True)

        if args.dry_run:
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "rc": 0,
                    "dry_run": True,
                    "wall_clock_s": 0.0,
                    "cmd": cmd,
                }
            )
            final_rc = 0
            break

        mode = "ab" if attempt_idx > 1 else "wb"
        with open(stdout_log, mode) as fout, open(stderr_log, mode) as ferr:
            header = f"\n===== attempt {attempt_idx} at {time.ctime()} =====\n".encode()
            fout.write(header)
            ferr.write(header)
            fout.flush()
            ferr.flush()
            try:
                rc = subprocess.call(cmd, stdout=fout, stderr=ferr)
            except FileNotFoundError as exc:
                rc = 127
                err_msg = f"[sweep_moe] {label}: launcher/script not found: {exc}\n"
                ferr.write(err_msg.encode())
                print(err_msg, file=driver_log, flush=True)

        wall = time.monotonic() - start
        attempts.append({"attempt": attempt_idx, "rc": rc, "wall_clock_s": wall, "cmd": cmd})
        msg = f"[sweep_moe] {label}: attempt {attempt_idx} -> rc={rc} wall={wall:.1f}s"
        print(msg, flush=True)
        print(msg, file=driver_log, flush=True)
        final_rc = rc

        if rc == 0:
            break
        if not _is_retryable_rc(rc):
            msg = f"[sweep_moe] {label}: rc={rc} is not retryable; giving up on this leaf."
            print(msg, flush=True)
            print(msg, file=driver_log, flush=True)
            break
        if attempt_idx >= max_attempts:
            msg = (
                f"[sweep_moe] {label}: exhausted retries ({args.per_leaf_max_retries}); giving up."
            )
            print(msg, flush=True)
            print(msg, file=driver_log, flush=True)
            break

        backoff = float(args.per_leaf_retry_backoff_s)
        msg = f"[sweep_moe] {label}: retryable rc={rc}; sleeping {backoff:.1f}s before resume."
        print(msg, flush=True)
        print(msg, file=driver_log, flush=True)
        time.sleep(backoff)

    stats = {
        "parallel_mode": parallel_mode,
        "num_tokens": num_tokens,
        "leaf_dir": str(leaf_dir),
        "final_rc": final_rc,
        "attempts": attempts,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _merge_results(output_dir: Path, leaf_stats: List[Dict[str, Any]]) -> Optional[Path]:
    """Concatenate per-leaf result.json into a single combined_result.json.

    Preserves per-leaf isolation (each leaf was its own python process) while
    giving downstream dashboards a single file to consume. ``leaves`` records
    every attempt so retry history is auditable.
    """
    combined: Dict[str, Any] = {
        "benchmark": "bench_moe (sweep_moe driver)",
        "leaves": [],
        "results": [],
        "rankings": [],
    }
    for stat in leaf_stats:
        leaf_dir = Path(stat["leaf_dir"])
        result_json = leaf_dir / "result.json"
        if not result_json.exists():
            combined["leaves"].append({**stat, "missing_result_json": True})
            continue
        try:
            with open(result_json) as f:
                payload = json.load(f)
        except Exception as exc:
            combined["leaves"].append({**stat, "load_error": f"{type(exc).__name__}: {exc}"})
            continue
        combined["leaves"].append(stat)
        combined["results"].extend(payload.get("results") or [])
        combined["rankings"].extend(payload.get("rankings") or [])
        # Carry forward environment / model / base_config / search from the
        # first leaf that has them; they are identical across leaves of one
        # sweep so first-write-wins is safe.
        for k in ("environment", "model", "base_config", "search"):
            if k not in combined and k in payload:
                combined[k] = payload[k]
    out_path = output_dir / "combined_result.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_driver_args(argv if argv is not None else sys.argv[1:])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        "launcher": args.launcher,
        "world_size": args.world_size,
        "parallel_modes": list(args.parallel_modes),
        "num_tokens": list(args.num_tokens),
        "bench_args": list(args.bench_args or []),
        "per_leaf_max_retries": args.per_leaf_max_retries,
        "per_leaf_retry_backoff_s": args.per_leaf_retry_backoff_s,
        "bench_script": args.bench_script,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    driver_log_path = output_dir / "driver.log"
    leaf_stats: List[Dict[str, Any]] = []
    sweep_start = time.monotonic()
    with open(driver_log_path, "w") as driver_log:
        print(f"[sweep_moe] start at {time.ctime()}", file=driver_log, flush=True)
        print(f"[sweep_moe] config: {json.dumps(sweep_config)}", file=driver_log, flush=True)
        for pm in args.parallel_modes:
            for nt in args.num_tokens:
                leaf_dir = _leaf_dir(output_dir, pm, nt)
                stat = _run_one_leaf(
                    args=args,
                    parallel_mode=pm,
                    num_tokens=nt,
                    leaf_dir=leaf_dir,
                    driver_log=driver_log,
                )
                leaf_stats.append(stat)
                if stat["final_rc"] != 0 and not args.continue_on_leaf_failure:
                    print(
                        f"[sweep_moe] leaf {pm} nt={nt} failed and "
                        "--continue_on_leaf_failure=False; aborting sweep.",
                        file=driver_log,
                        flush=True,
                    )
                    break
        sweep_wall = time.monotonic() - sweep_start
        print(
            f"[sweep_moe] all leaves done in {sweep_wall:.1f}s",
            file=driver_log,
            flush=True,
        )

    success_leaves = sum(1 for s in leaf_stats if s["final_rc"] == 0)
    fail_leaves = len(leaf_stats) - success_leaves
    print(
        f"[sweep_moe] {success_leaves}/{len(leaf_stats)} leaves OK, {fail_leaves} failed.",
        flush=True,
    )
    print(f"[sweep_moe] driver log: {driver_log_path}", flush=True)

    if not args.no_combined_report:
        combined = _merge_results(output_dir, leaf_stats)
        if combined is not None:
            print(f"[sweep_moe] combined report: {combined}", flush=True)

    return 0 if fail_leaves == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
