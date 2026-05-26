from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument("--step-limit", type=int, default=12)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--cost-tracking", default=os.getenv("MSWEA_COST_TRACKING", "ignore_errors"))
    parser.add_argument("--timeout", type=int, default=60)
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
        default=os.getenv("MSWEA_RUNTIME_INPUT", "10"),
        help="Input passed to agents during run/verify rounds",
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
    return parser.parse_args()


def default_state_output(output_path: str) -> str:
    if output_path.endswith(".traj.json"):
        return output_path.removesuffix(".traj.json") + ".state.json"
    return output_path + ".state.json"


def save_state(path: str, state: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


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
        }
    )

    state_output = args.state_output or default_state_output(args.output)
    if state_output:
        save_state(state_output, result)
        result["_state_output"] = state_output

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
