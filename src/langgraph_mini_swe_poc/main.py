from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv

from langgraph_mini_swe_poc.graph import build_graph


def default_api_base_for_model(model: str) -> str:
    env_api_base = os.getenv("LITELLM_API_BASE", "")
    if env_api_base:
        return env_api_base
    if "/" not in model and not model.startswith(("gpt-", "o1", "o3", "o4")):
        return "http://localhost:11436"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LangGraph + mini-SWE-agent PoC.")
    parser.add_argument("task", nargs="?", help="Task for the coding worker")
    parser.add_argument("--model", default=os.getenv("MSWEA_MODEL_NAME", "qwen3:32b"))
    parser.add_argument("--image", default="python:3.11", help="Docker image for the worker sandbox")
    parser.add_argument("--cwd", default="/tmp", help="Working directory inside the sandbox")
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--cost-tracking", default=os.getenv("MSWEA_COST_TRACKING", "ignore_errors"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--api-base", default=None)
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "negotiate", "dual-swe", "loop"],
        default="single",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.getenv("MSWEA_ROUNDS", "3")),
        help="Number of coordinator rounds for loop mode",
    )
    parser.add_argument(
        "--agents",
        default=os.getenv("MSWEA_AGENT_IDS", "agent_a,agent_b,agent_c"),
        help="Comma-separated agent IDs for loop mode",
    )
    parser.add_argument(
        "--runtime-input",
        default=os.getenv("MSWEA_RUNTIME_INPUT", "30"),
        help="Input n passed to agents when running their Fibonacci implementation",
    )
    parser.add_argument(
        "--expected-value",
        default=os.getenv("MSWEA_EXPECTED_VALUE", "832040"),
        help="Expected output for fib(runtime-input); agents assert against this value",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MSWEA_TRAJECTORY_PATH", "trajectories/last_run.traj.json"),
        help="Path for the mini-SWE-agent trajectory JSON file",
    )
    parser.add_argument(
        "--state-output",
        default=os.getenv("LANGGRAPH_STATE_PATH", ""),
        help="Path for the full LangGraph final state JSON. Defaults to --output with .state.json suffix.",
    )
    parser.add_argument(
        "--shared-dir",
        default=os.getenv("MSWEA_SHARED_DIR", ""),
        help="Optional host directory mounted into containers; empty means no shared filesystem",
    )
    parser.add_argument("--shared-mount", default="/workspace/shared")
    parser.add_argument("--graph", action="store_true", help="Print the graph as Mermaid and exit")
    parser.add_argument("--mock", action="store_true", help="Run graph wiring without LLM/Docker")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full state JSON instead of compact summary")
    return parser.parse_args()


def default_state_output(output_path: str) -> str:
    if output_path.endswith(".traj.json"):
        return output_path.removesuffix(".traj.json") + ".state.json"
    return output_path + ".state.json"


def save_state(path: str, state: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def _traj_cost(traj_path: str) -> tuple[float, int]:
    """Return (instance_cost, api_calls) summed across all round trajectories for this agent.

    The stored path is for the last round only (e.g. ``agent_a.r3.traj.json``).
    We glob sibling files to pick up every round (``agent_a.r*.traj.json``).
    """
    p = Path(traj_path)
    # Strip the round suffix to build a glob: "…agent_a.r3.traj.json" → "…agent_a.r*.traj.json"
    stem = p.name  # e.g. "loop-openai-fib.agent_a.r3.traj.json"
    parts = stem.split(".")
    # Find the round segment (starts with "r" followed by digits)
    for i, part in enumerate(parts):
        if part.startswith("r") and part[1:].isdigit():
            parts[i] = "r*"
            break
    pattern = ".".join(parts)
    total_cost, total_calls = 0.0, 0
    for sibling in sorted(p.parent.glob(pattern)):
        try:
            data = json.loads(sibling.read_text())
            stats = data.get("info", {}).get("model_stats", {})
            total_cost += stats.get("instance_cost", 0.0)
            total_calls += stats.get("api_calls", 0)
        except Exception:
            pass
    return total_cost, total_calls


def _read_test_outcome(shared_dir: str, agent_id: str) -> str:
    """Read the test log from disk and return a one-line outcome string.

    Returns an empty string when no log file exists (caller falls back to
    submission text).  Never silently swallows a failure.
    """
    if not shared_dir:
        return ""
    log_path = Path(shared_dir) / f"test_{agent_id}.log"
    if not log_path.exists():
        return "no test log"
    content = log_path.read_text().strip()
    if not content:
        return "test log empty (tests may not have run)"
    # Pytest summary line is the last non-empty line, e.g.:
    #   "5 passed in 0.12s"  /  "1 failed, 4 passed in 0.15s"  /  "ERROR ..."
    last_line = content.splitlines()[-1].strip()
    lower = content.lower()
    if "no module named pytest" in lower or "modulenotfounderror" in lower:
        return f"FAIL (pytest not installed): {last_line}"
    if "failed" in lower or "error" in lower:
        return f"FAIL: {last_line}"
    return f"PASS: {last_line}"


def _verify_run_output(shared_dir: str, agent_id: str, runtime_input: str) -> str:
    """Harness-side check: execute fib_agent_*.py directly and verify the output.

    The harness runs the implementation itself — it never reads the agent-written
    run log, so agents cannot cheat by writing a fake value to that file.
    Returns empty string on pass, error message on fail.
    """
    if not shared_dir or not runtime_input:
        return ""

    script = Path(shared_dir) / f"fib_{agent_id}.py"
    if not script.exists():
        return f"HARNESS: {script.name} not found"

    # Compute ground truth independently in the harness process.
    a, b = 0, 1
    for _ in range(int(runtime_input)):
        a, b = b, a + b
    true_value = str(a)

    # Execute the agent's implementation directly.
    try:
        proc = subprocess.run(
            [sys.executable, str(script), runtime_input],
            capture_output=True, text=True, timeout=10,
        )
        actual = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"HARNESS: fib_{agent_id}.py timed out"
    except Exception as exc:
        return f"HARNESS: failed to run fib_{agent_id}.py: {exc}"

    # Overwrite the run log with the harness-executed result.
    # The agent-written value is discarded; this log is now the authoritative record.
    run_log = Path(shared_dir) / f"run_{agent_id}.log"
    run_log.write_text(actual + "\n")

    if actual != true_value:
        return f"HARNESS ASSERT FAILED: fib({runtime_input})={true_value}, got {actual!r}"
    return ""


def _print_summary(result: dict, state_output: str) -> None:
    """Print a compact run summary instead of the full state JSON."""
    mode = result.get("mode", "?")
    print(f"\n=== Run complete (mode={mode}) ===")

    grand_cost = 0.0

    # Loop mode: per-agent results table
    final = result.get("final", {})
    agent_results = final.get("agent_results", {})
    shared_dir = result.get("shared_dir", "")
    runtime_input = result.get("runtime_input", "")
    if agent_results:
        for agent_id, res in sorted(agent_results.items()):
            status = res.get("exit_status", "?")
            traj = res.get("trajectory_path", "")

            # Harness-side verification: ground truth computed from runtime_input,
            # never from expected_value — agents cannot game this by hardcoding it.
            harness_fail = _verify_run_output(shared_dir, agent_id, runtime_input)

            # Test log outcome from disk.
            test_outcome = _read_test_outcome(shared_dir, agent_id)

            if harness_fail:
                outcome = f"  → {harness_fail}"
            elif test_outcome:
                outcome = f"  → {test_outcome}"
            elif status == "Submitted":
                outcome = "  → OK"
            else:
                submission = res.get("submission", "")
                last_line = submission.strip().splitlines()[-1] if submission.strip() else ""
                outcome = f"  → {last_line}" if last_line else ""

            cost, calls = _traj_cost(traj) if traj else (0.0, 0)
            grand_cost += cost
            cost_str = f"  ${cost:.4f} / {calls} calls" if calls else ""
            print(f"  {agent_id:12s}  {status}{outcome}")
            if cost_str:
                print(f"  {'':12s}{cost_str}")
    else:
        # Single/multi/other modes
        worker = result.get("worker_result") or {}
        status = worker.get("exit_status", result.get("exit_status", "?"))
        submission = worker.get("submission", "")
        last_line = submission.strip().splitlines()[-1] if submission.strip() else ""
        traj = worker.get("trajectory_path", "")
        cost, calls = _traj_cost(traj) if traj else (0.0, 0)
        grand_cost += cost
        print(f"  status: {status}")
        if last_line:
            print(f"  → {last_line}")
        if calls:
            print(f"  cost:   ${cost:.4f}  ({calls} API calls)")

    if grand_cost:
        print(f"\n  Total cost: ${grand_cost:.4f}")
    print(f"  Full state: {state_output}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_base = args.api_base
    if api_base is None:
        api_base = default_api_base_for_model(args.model)

    graph = build_graph(args.mode)
    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return

    if not args.task:
        raise SystemExit("error: task is required unless --graph is used")

    agent_ids = [item.strip() for item in args.agents.split(",") if item.strip()]
    result = graph.invoke(
        {
            "task": args.task,
            "mode": args.mode,
            "model": args.model,
            "image": args.image,
            "cwd": args.cwd,
            "step_limit": args.step_limit,
            "cost_limit": args.cost_limit,
            "cost_tracking": args.cost_tracking,
            "timeout": args.timeout,
            "api_base": api_base or None,
            "output_path": args.output,
            "shared_dir": str(Path(args.shared_dir).expanduser().resolve()) if args.shared_dir else "",
            "shared_mount": args.shared_mount,
            "mock": args.mock,
            "round_limit": args.rounds,
            "agent_ids": agent_ids,
            "runtime_input": args.runtime_input,
            "expected_value": args.expected_value,
        }
    )

    state_output = args.state_output or default_state_output(args.output)
    if state_output:
        save_state(state_output, result)
        result["_state_output"] = state_output

    if args.verbose:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result, state_output)


if __name__ == "__main__":
    main()
