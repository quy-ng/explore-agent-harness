from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any, TypedDict
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, START, StateGraph

from langgraph_mini_swe_poc.chat import chat_once
from langgraph_mini_swe_poc.worker import run_swe_worker


class PocState(TypedDict, total=False):
    task: str
    input: str
    messages: list[dict[str, Any]]
    mode: str
    worker_task: str
    verifier_task: str
    artifact_report: dict[str, Any]
    planner_message: dict[str, Any]
    plan_review_message: dict[str, Any]
    revised_planner_message: dict[str, Any]
    reviewer_message: dict[str, Any]
    conversation: list[dict[str, Any]]
    model: str
    image: str
    cwd: str
    step_limit: int
    cost_limit: float
    cost_tracking: str
    timeout: int
    api_base: str | None
    output_path: str
    shared_dir: str
    shared_mount: str
    mock: bool
    worker_result: dict[str, Any]
    implementer_result: dict[str, Any]
    verifier_result: dict[str, Any]
    round: int
    round_limit: int
    agent_ids: list[str]
    runtime_input: str
    expected_value: str
    agents: dict[str, Any]
    coordinator_tasks: dict[str, str]
    agent_results: dict[str, Any]
    final: dict[str, Any]


def _extract_task(state: PocState) -> str:
    task = state.get("task") or state.get("input") or ""
    if task:
        return str(task).strip()
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        content = last.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
    return ""


def normalize_input(state: PocState) -> PocState:
    task = _extract_task(state)
    if not task:
        task = "Create /tmp/hello.py that prints hello, run it, then submit the result"
    return {
        "task": task,
        "mode": state.get("mode", "single"),
        "model": state.get("model", "openai/gpt-4.1"),
        "image": state.get("image", "python:3.11"),
        "cwd": state.get("cwd", "/tmp"),
        "step_limit": state.get("step_limit", 12),
        "cost_limit": state.get("cost_limit", 1.0),
        "cost_tracking": state.get("cost_tracking", "ignore_errors"),
        "timeout": state.get("timeout", 60),
        "api_base": state.get("api_base"),
        "output_path": state.get("output_path", "trajectories/studio-run.traj.json"),
        "shared_dir": state.get("shared_dir", ""),
        "shared_mount": state.get("shared_mount", "/workspace/shared"),
        "mock": state.get("mock", False),
        "round": state.get("round", 1),
        "round_limit": state.get("round_limit", 3),
        "agent_ids": state.get("agent_ids", ["agent_a", "agent_b", "agent_c"]),
        "runtime_input": str(state.get("runtime_input", "12")),
        "expected_value": str(state.get("expected_value", "144")),
        "agents": state.get("agents", {}),
    }


def _test_run_cmd(agent_id: str) -> str:
    """Return the canonical command to run an agent's test file.

    The ``run_tests`` executable is injected into ``/usr/local/bin`` inside the
    container by ``_docker_args`` so the agent only needs to call it by name.
    It handles ``cd`` into the shared directory itself, ensuring relative imports
    (e.g. ``from fib_agent_a import fib``) always resolve correctly — even if the
    agent forgets to change directory first.
    """
    return f"run_tests {agent_id}"


_RUN_TESTS_SCRIPT = """\
#!/bin/sh
# Usage: run_tests <agent_id>
# Canonical test runner injected by the harness.
# - cd into the shared directory so relative imports (from fib_agent_X import ...) resolve.
# - Installs pytest if absent (python:3.x images ship stdlib only).
# - set -o pipefail ensures a test failure propagates through the tee pipeline.
set -eo pipefail
AGENT_ID="${1:?Usage: run_tests <agent_id>}"
SHARED="${SHARED_MOUNT:-/workspace/shared}"
LOG="$SHARED/test_$AGENT_ID.log"
cd "$SHARED"
python3 -m pytest --version >/dev/null 2>&1 || python3 -m pip install pytest -q --disable-pip-version-check
python3 -m pytest "test_fib_$AGENT_ID.py" -v 2>&1 | tee "$LOG"
"""


def _default_agent_task(
    round_index: int,
    agent_id: str,
    runtime_input: str,
    expected_value: str,
    shared_mount: str,
) -> str:
    test_cmd = _test_run_cmd(agent_id)
    fib_script = f"{shared_mount}/fib_{agent_id}.py"
    if round_index == 1:
        approach = {
            "agent_a": "recursive",
            "agent_b": "matrix exponentiation",
            "agent_c": "fast doubling",
        }.get(agent_id, "iterative")
        return (
            "Implement Fibonacci with the approach below. Use a unique file name like "
            f"`fib_{agent_id}.py`. Provide a small CLI or main guard so it can be run.\n\n"
            f"Approach: {approach}.\n\n"
            "Expected outputs:\n"
            f"- fib_{agent_id}.py\n"
        )
    if round_index == 2:
        return (
            f"Add tests for your Fibonacci implementation at {fib_script}. "
            f"Use a separate test file such as `test_fib_{agent_id}.py`.\n\n"
            f"Run tests with exactly: `{test_cmd} > test_{agent_id}.log 2>&1`\n\n"
            "Expected outputs:\n"
            f"- test_fib_{agent_id}.py\n"
            f"- test_{agent_id}.log"
        )
    if round_index == 3:
        step_a = "\n".join([
            # Condition 1: output must match expected value
            f'result=$(python3 {fib_script} {runtime_input}) && echo "$result" > {shared_mount}/run_{agent_id}.log &&',
            f'[ "$result" = "{expected_value}" ] || {{ echo "CONDITION 1 FAILED: fib({runtime_input})=$result, expected {expected_value}"; exit 1; }} &&',
            f'echo "CONDITION 1 PASSED: fib({runtime_input})={expected_value}" &&',
            # Condition 2: all unit tests must pass
            f"{test_cmd} &&",
            'echo "CONDITION 2 PASSED: all tests passed" &&',
            # Print evidence
            f"cat {fib_script} &&",
            f"cat {shared_mount}/run_{agent_id}.log &&",
            f"cat {shared_mount}/test_{agent_id}.log",
        ])
        return (
            "Final round: copy and run the exact commands below — do NOT change any values.\n"
            "Both conditions must pass before you may submit:\n"
            f"  • Condition 1: fib({runtime_input}) must equal {expected_value}\n"
            "  • Condition 2: all unit tests must pass\n\n"
            f"Step A (copy exactly):\n```\n{step_a}\n```\n\n"
            "Step B — only after Step A exits 0 with both conditions passing, run this single command alone:\n"
            "```\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```\n\n"
            "Expected outputs:\n"
            f"- {shared_mount}/run_{agent_id}.log\n"
            f"- {shared_mount}/test_{agent_id}.log"
        )
    return (
        "Repeat a validation run on your implementation. Use a fresh input, show the command, "
        "and include logs proving execution."
    )


def coordinator_round(state: PocState) -> PocState:
    round_index = int(state.get("round", 1))
    runtime_input = str(state.get("runtime_input"))
    expected_value = str(state.get("expected_value"))
    narrative = str(state.get("task")).strip()
    shared_dir = str(state.get("shared_dir", "")).strip()
    shared_mount = str(state.get("shared_mount")).strip()
    shared_note = ""
    if shared_dir:
        shared_note = (
            "\n\nShared output: write all code, tests, and logs under "
            f"`{shared_mount}` so they are available on the host."
        )
    tasks: dict[str, str] = {}
    for agent_id in state.get("agent_ids", []):
        base_task = _default_agent_task(round_index, agent_id, runtime_input, expected_value, shared_mount)
        if narrative:
            base_task = f"{base_task}\n\nCoordinator narrative:\n{narrative}"
        if shared_note:
            base_task = f"{base_task}{shared_note}"
        tasks[agent_id] = base_task
    return {"coordinator_tasks": tasks}


def _agent_prompt(agent_id: str, task: str, history: list[dict[str, Any]]) -> str:
    history_text = ""
    if history:
        trimmed = history[-5:]
        rendered = [
            f"Round {item.get('round')}: {item.get('summary', '')}" for item in trimmed
        ]
        history_text = "\n".join(line for line in rendered if line)
    return (
        f"You are {agent_id}, an isolated coding agent. "
        "Do not reference or reuse any other agent's work. "
        "Only use your own history below.\n\n"
        f"History:\n{history_text or 'None'}\n\n"
        f"Task:\n{task}\n\n"
        "Keep the change minimal, verify it with a command, then submit a concise summary."
    )


def _agent_round_output_path(state: PocState, agent_id: str, round_index: int) -> str:
    output_path = state.get("output_path", "trajectories/last_run.traj.json")
    path = output_path.rsplit(".traj.json", 1)[0]
    if path == output_path:
        return f"{output_path}.{agent_id}.r{round_index}.traj.json"
    return f"{path}.{agent_id}.r{round_index}.traj.json"


def run_agents(state: PocState) -> PocState:
    round_index = int(state.get("round", 1))
    tasks = state.get("coordinator_tasks", {})
    agents_state = dict(state.get("agents", {}))
    results: dict[str, Any] = {}

    def _summarize_trajectory(trajectory_path: str) -> str:
        if not trajectory_path or not Path(trajectory_path).is_file():
            return ""
        data = json.loads(Path(trajectory_path).read_text())
        messages = data.get("messages", [])
        commands: list[str] = []
        outputs: list[str] = []
        for message in messages:
            if message.get("role") == "assistant":
                for action in message.get("extra", {}).get("actions", []):
                    command = str(action.get("command", "")).strip()
                    if command:
                        commands.append(command)
            if message.get("role") == "tool":
                output = message.get("extra", {}).get("raw_output", "")
                if output:
                    outputs.append(str(output).strip())
        commands = commands[-5:]
        outputs = outputs[-3:]
        summary_lines = []
        if commands:
            summary_lines.append("Commands:")
            summary_lines.extend(f"- {command}" for command in commands)
        if outputs:
            summary_lines.append("Outputs:")
            summary_lines.extend(f"- {output}" for output in outputs)
        return "\n".join(summary_lines).strip()

    def _run_agent(agent_id: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        agent_state = dict(agents_state.get(agent_id, {}))
        history = list(agent_state.get("history", []))
        prompt = _agent_prompt(agent_id, tasks.get(agent_id, ""), history)
        result = run_swe_worker(
            task=prompt,
            model=state["model"],
            image=state.get("image", "python:3.11"),
            cwd=state.get("cwd", "/tmp"),
            step_limit=state.get("step_limit", 12),
            cost_limit=state.get("cost_limit", 1.0),
            cost_tracking=state.get("cost_tracking", "ignore_errors"),
            timeout=state.get("timeout", 60),
            api_base=state.get("api_base"),
            output_path=_agent_round_output_path(state, agent_id, round_index),
            docker_args=_docker_args(state),
            mock=state.get("mock", False),
        )
        if not result.get("submission"):
            summary = _summarize_trajectory(result.get("trajectory_path", ""))
            if summary:
                result["submission"] = summary
        summary = {
            "round": round_index,
            "summary": (
                f"task={tasks.get(agent_id, '')!r} "
                f"exit_status={result.get('exit_status', '')} "
                f"submission={result.get('submission', '')!r}"
            ),
            "result": result,
        }
        history.append(summary)
        return agent_id, result, history

    agent_ids = list(state.get("agent_ids", []))
    with ThreadPoolExecutor(max_workers=len(agent_ids) or None) as executor:
        futures = [executor.submit(_run_agent, agent_id) for agent_id in agent_ids]
        for future in futures:
            agent_id, result, history = future.result()
            agent_state = dict(agents_state.get(agent_id, {}))
            agent_state["history"] = history
            agent_state["last_result"] = result
            agents_state[agent_id] = agent_state
            results[agent_id] = result

    return {"agents": agents_state, "agent_results": results}


def advance_round(state: PocState) -> PocState:
    return {"round": int(state.get("round", 1)) + 1}


def should_continue(state: PocState) -> str:
    round_index = int(state.get("round", 1))
    round_limit = int(state.get("round_limit", 1))
    return "continue" if round_index <= round_limit else "stop"


def finalize_loop(state: PocState) -> PocState:
    final = {
        "round": state.get("round", 1),
        "round_limit": state.get("round_limit", 1),
        "agent_ids": state.get("agent_ids", []),
        "agents": state.get("agents", {}),
        "agent_results": state.get("agent_results", {}),
        "shared_dir": state.get("shared_dir", ""),
        "shared_mount": state.get("shared_mount", ""),
    }
    return {"final": final}


def planner_agent(state: PocState) -> PocState:
    message = chat_once(
        role="planner",
        prompt=(
            "Plan a minimal implementation for this coding task. "
            "Give the SWE worker concrete steps and verification advice.\n\n"
            f"Task: {state['task']}"
        ),
        model=state["model"],
        api_base=state.get("api_base"),
        mock=state.get("mock", False),
    )
    return {"planner_message": message, "conversation": [message]}


def plan_reviewer_agent(state: PocState) -> PocState:
    message = chat_once(
        role="plan_reviewer",
        prompt=(
            "Review this proposed plan before any SWE worker executes it. "
            "Look for missing verification steps, ambiguous file paths, unsafe assumptions, "
            "and places where the worker may fail. Give concise critique and concrete revisions.\n\n"
            f"Original task:\n{state['task']}\n\n"
            f"Planner proposal:\n{state.get('planner_message', {}).get('content', '')}"
        ),
        model=state["model"],
        api_base=state.get("api_base"),
        mock=state.get("mock", False),
    )
    return {"plan_review_message": message, "conversation": [*state.get("conversation", []), message]}


def revise_plan_agent(state: PocState) -> PocState:
    message = chat_once(
        role="planner",
        prompt=(
            "Revise the plan using the reviewer critique. Produce the final instructions for the SWE worker. "
            "Be concrete and include verification commands.\n\n"
            f"Original task:\n{state['task']}\n\n"
            f"Initial plan:\n{state.get('planner_message', {}).get('content', '')}\n\n"
            f"Reviewer critique:\n{state.get('plan_review_message', {}).get('content', '')}"
        ),
        model=state["model"],
        api_base=state.get("api_base"),
        mock=state.get("mock", False),
    )
    return {
        "planner_message": message,
        "revised_planner_message": message,
        "conversation": [*state.get("conversation", []), message],
    }


def prepare_task(state: PocState) -> PocState:
    task = state["task"].strip()
    planner_content = state.get("planner_message", {}).get("content")
    planner_section = f"\n\nPlanner guidance:\n{planner_content}" if planner_content else ""
    dual_section = ""
    if state.get("mode") == "dual-swe":
        dual_section = (
            "\n\nDual-SWE artifact handoff contract:\n"
            "- Do not rely on a shared filesystem with the verifier.\n"
            "- If you create `print_ip.py`, create it in your sandbox, verify it, then submit its content as text.\n"
            "- Use this exact final submit pattern, replacing the path if needed:\n"
            "  `printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nARTIFACT_PATH=print_ip.py\\n"
            "ARTIFACT_CONTENT_BASE64=' && base64 -w0 /tmp/print_ip.py && printf '\\n'`\n"
            "- The verifier will reconstruct the file from your submitted string."
        )
    worker_task = (
        f"{task}\n\n"
        "Keep the change small. Verify it by running the relevant command. "
        "When done, submit a concise summary."
        f"{planner_section}"
        f"{dual_section}"
    )
    return {"worker_task": worker_task}


def _docker_args(state: PocState) -> list[str]:
    shared_dir = state.get("shared_dir")
    shared_mount = state.get("shared_mount", "/workspace/shared")
    if not shared_dir:
        return []
    host_dir = Path(shared_dir).expanduser().resolve()
    host_dir.mkdir(parents=True, exist_ok=True)

    # Write the canonical test-runner script to the shared dir on the host.
    # It is mounted read-only into /usr/local/bin so all agents find it on $PATH.
    script_path = host_dir / "run_tests"
    script_path.write_text(_RUN_TESTS_SCRIPT)
    script_path.chmod(0o755)

    return [
        "-v", f"{host_dir}:{shared_mount}",
        "-v", f"{script_path}:/usr/local/bin/run_tests:ro",
        "-e", f"SHARED_MOUNT={shared_mount}",
    ]


def run_worker(state: PocState) -> PocState:
    result = run_swe_worker(
        task=state["worker_task"],
        model=state["model"],
        image=state.get("image", "python:3.11"),
        cwd=state.get("cwd", "/tmp"),
        step_limit=state.get("step_limit", 12),
        cost_limit=state.get("cost_limit", 1.0),
        cost_tracking=state.get("cost_tracking", "ignore_errors"),
        timeout=state.get("timeout", 60),
        api_base=state.get("api_base"),
        output_path=state.get("output_path", "trajectories/last_run.traj.json"),
        docker_args=_docker_args(state),
        mock=state.get("mock", False),
    )
    return {"worker_result": result}


def _agent_output_path(state: PocState, label: str) -> str:
    output_path = state.get("output_path", "trajectories/last_run.traj.json")
    path = output_path.rsplit(".traj.json", 1)[0]
    if path == output_path:
        return f"{output_path}.{label}.traj.json"
    return f"{path}.{label}.traj.json"


def run_implementer_swe(state: PocState) -> PocState:
    result = run_swe_worker(
        task=state["worker_task"],
        model=state["model"],
        image=state.get("image", "python:3.11"),
        cwd=state.get("cwd", "/tmp"),
        step_limit=state.get("step_limit", 12),
        cost_limit=state.get("cost_limit", 1.0),
        cost_tracking=state.get("cost_tracking", "ignore_errors"),
        timeout=state.get("timeout", 60),
        api_base=state.get("api_base"),
        output_path=_agent_output_path(state, "implementer"),
        docker_args=_docker_args(state),
        mock=state.get("mock", False),
    )
    message = {
        "role": "swe_implementer",
        "content": (
            f"Implementer exit_status={result.get('exit_status')} "
            f"submission={result.get('submission')!r} "
            f"trajectory={result.get('trajectory_path')}"
        ),
        "result": result,
    }
    return {
        "implementer_result": result,
        "worker_result": result,
        "conversation": [*state.get("conversation", []), message],
    }


def _extract_artifact_report(state: PocState) -> dict[str, Any]:
    result = state["implementer_result"]
    submission = result.get("submission", "") or ""
    trajectory_path = result.get("trajectory_path", "")

    if not submission and trajectory_path and Path(trajectory_path).is_file():
        data = json.loads(Path(trajectory_path).read_text())
        submission = data.get("info", {}).get("submission", "") or ""

    path_match = re.search(r"^ARTIFACT_PATH=([^\r\n]+)$", submission, re.MULTILINE)
    b64_match = re.search(r"^ARTIFACT_CONTENT_BASE64=([A-Za-z0-9+/=]+)$", submission, re.MULTILINE)
    artifact_path = path_match.group(1).strip() if path_match else "print_ip.py"
    artifact_content = ""
    error = ""

    if b64_match:
        payload = re.sub(r"\s+", "", b64_match.group(1))
        try:
            artifact_content = base64.b64decode(payload).decode("utf-8")
        except Exception as exc:
            error = f"Could not decode ARTIFACT_CONTENT_BASE64: {exc}"
    else:
        error = "Implementer submission did not include ARTIFACT_CONTENT_BASE64."

    return {
        "artifact_path": artifact_path,
        "artifact_content": artifact_content,
        "raw_submission": submission,
        "source_trajectory": trajectory_path,
        "error": error,
    }


def collect_implementer_artifact(state: PocState) -> PocState:
    report = _extract_artifact_report(state)
    message = {
        "role": "artifact_collector",
        "content": (
            f"Collected artifact_path={report.get('artifact_path')!r}, "
            f"content_length={len(report.get('artifact_content', ''))}, "
            f"error={report.get('error')!r}"
        ),
        "artifact_report": report,
    }
    return {"artifact_report": report, "conversation": [*state.get("conversation", []), message]}


def prepare_verifier_task(state: PocState) -> PocState:
    implementer = state["implementer_result"]
    planner = state.get("planner_message", {}).get("content", "")
    artifact = state.get("artifact_report", {})
    artifact_path = artifact.get("artifact_path", "print_ip.py")
    artifact_content = artifact.get("artifact_content", "")
    verifier_task = (
        "You are a second SWE agent acting as an independent verifier in a fresh sandbox. "
        "You cannot access the first SWE agent's filesystem. LangGraph passed you the artifact content as text. "
        f"Write that exact string to `/tmp/{artifact_path}`, then run `python /tmp/{artifact_path}`. "
        "Also call `curl -s https://api.ipify.org` from your verifier sandbox. "
        "Compare the output of the Python file with the ipify result. "
        "Verification passes only if both outputs are the same IPv4 address. "
        "If they differ, or either value is not an IPv4 address, raise the issue in your final submission. "
        "Finish with the submit command when done.\n\n"
        f"Original task:\n{state['task']}\n\n"
        f"Planner guidance:\n{planner}\n\n"
        f"Implementer result:\n{implementer}\n"
        f"Artifact collection report:\n{artifact}\n\n"
        f"Artifact content for `/tmp/{artifact_path}`:\n```python\n{artifact_content}\n```\n"
    )
    return {"verifier_task": verifier_task}


def run_verifier_swe(state: PocState) -> PocState:
    result = run_swe_worker(
        task=state["verifier_task"],
        model=state["model"],
        image=state.get("image", "python:3.11"),
        cwd=state.get("cwd", "/tmp"),
        step_limit=state.get("step_limit", 12),
        cost_limit=state.get("cost_limit", 1.0),
        cost_tracking=state.get("cost_tracking", "ignore_errors"),
        timeout=state.get("timeout", 60),
        api_base=state.get("api_base"),
        output_path=_agent_output_path(state, "verifier"),
        docker_args=_docker_args(state),
        mock=state.get("mock", False),
    )
    message = {
        "role": "swe_verifier",
        "content": (
            f"Verifier exit_status={result.get('exit_status')} "
            f"submission={result.get('submission')!r} "
            f"trajectory={result.get('trajectory_path')}"
        ),
        "result": result,
    }
    return {
        "verifier_result": result,
        "conversation": [*state.get("conversation", []), message],
    }


def reviewer_agent(state: PocState) -> PocState:
    worker_result = state.get("verifier_result") or state["worker_result"]
    implementer_result = state.get("implementer_result")
    message = chat_once(
        role="reviewer",
        prompt=(
            "Review this SWE worker result. Say whether it appears successful, "
            "what evidence supports that, and what the next move should be.\n\n"
            f"Original task: {state['task']}\n\n"
            f"Planner said: {state.get('planner_message', {}).get('content', '')}\n\n"
            f"Implementer result: {implementer_result}\n\n"
            f"Worker result: {worker_result}"
        ),
        model=state["model"],
        api_base=state.get("api_base"),
        mock=state.get("mock", False),
    )
    conversation = [*state.get("conversation", []), message]
    return {"reviewer_message": message, "conversation": conversation}


def review_result(state: PocState) -> PocState:
    result = state.get("verifier_result") or state["worker_result"]
    final = {
        "task": state["task"],
        "exit_status": result.get("exit_status", "unknown"),
        "submission": result.get("submission", ""),
        "conversation": state.get("conversation", []),
        "review": state.get("reviewer_message", {}),
        "implementer": state.get("implementer_result", {}),
        "artifact_report": state.get("artifact_report", {}),
        "verifier": state.get("verifier_result", {}),
        "shared_dir": state.get("shared_dir", ""),
        "shared_mount": state.get("shared_mount", ""),
        "worker": result,
    }
    return {"final": final}


def _add_single_nodes(graph: StateGraph) -> None:
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_worker", run_worker)
    graph.add_node("review_result", review_result)


def _add_multi_nodes(graph: StateGraph) -> None:
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_worker", run_worker)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_node("review_result", review_result)


def _add_negotiate_nodes(graph: StateGraph) -> None:
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("plan_reviewer_agent", plan_reviewer_agent)
    graph.add_node("revise_plan_agent", revise_plan_agent)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_worker", run_worker)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_node("review_result", review_result)


def _add_dual_swe_nodes(graph: StateGraph) -> None:
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_implementer_swe", run_implementer_swe)
    graph.add_node("collect_implementer_artifact", collect_implementer_artifact)
    graph.add_node("prepare_verifier_task", prepare_verifier_task)
    graph.add_node("run_verifier_swe", run_verifier_swe)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_node("review_result", review_result)


def _add_loop_nodes(graph: StateGraph) -> None:
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("coordinator_round", coordinator_round)
    graph.add_node("run_agents", run_agents)
    graph.add_node("advance_round", advance_round)
    graph.add_node("finalize_loop", finalize_loop)


def _wire_entry(graph: StateGraph, mode: str) -> None:
    graph.add_edge(START, "normalize_input")
    if mode in {"multi", "negotiate", "dual-swe"}:
        graph.add_edge("normalize_input", "planner_agent")
        if mode == "negotiate":
            graph.add_node("plan_reviewer_agent", plan_reviewer_agent)
            graph.add_node("revise_plan_agent", revise_plan_agent)
            graph.add_edge("planner_agent", "plan_reviewer_agent")
            graph.add_edge("plan_reviewer_agent", "revise_plan_agent")
            graph.add_edge("revise_plan_agent", "prepare_task")
        else:
            graph.add_edge("planner_agent", "prepare_task")
    elif mode == "loop":
        graph.add_edge("normalize_input", "coordinator_round")
    else:
        graph.add_edge("normalize_input", "prepare_task")


def _wire_execution(graph: StateGraph, mode: str) -> None:
    if mode == "dual-swe":
        graph.add_edge("prepare_task", "run_implementer_swe")
        graph.add_edge("run_implementer_swe", "collect_implementer_artifact")
        graph.add_edge("collect_implementer_artifact", "prepare_verifier_task")
        graph.add_edge("prepare_verifier_task", "run_verifier_swe")
        graph.add_edge("run_verifier_swe", "reviewer_agent")
        graph.add_edge("reviewer_agent", "review_result")
    elif mode == "loop":
        graph.add_edge("coordinator_round", "run_agents")
        graph.add_edge("run_agents", "advance_round")
        graph.add_conditional_edges(
            "advance_round",
            should_continue,
            {"continue": "coordinator_round", "stop": "finalize_loop"},
        )
        graph.add_edge("finalize_loop", END)
    else:
        graph.add_edge("prepare_task", "run_worker")


def _wire_review(graph: StateGraph, mode: str) -> None:
    if mode in {"multi", "negotiate"}:
        graph.add_edge("run_worker", "reviewer_agent")
        graph.add_edge("reviewer_agent", "review_result")
    elif mode == "single":
        graph.add_edge("run_worker", "review_result")
    graph.add_edge("review_result", END)


def build_single_graph():
    graph = StateGraph(PocState)
    _add_single_nodes(graph)
    _wire_entry(graph, "single")
    _wire_execution(graph, "single")
    _wire_review(graph, "single")
    return graph.compile()


def build_multi_graph():
    graph = StateGraph(PocState)
    _add_multi_nodes(graph)
    _wire_entry(graph, "multi")
    _wire_execution(graph, "multi")
    _wire_review(graph, "multi")
    return graph.compile()


def build_negotiate_graph():
    graph = StateGraph(PocState)
    _add_negotiate_nodes(graph)
    _wire_entry(graph, "negotiate")
    _wire_execution(graph, "negotiate")
    _wire_review(graph, "negotiate")
    return graph.compile()


def build_dual_swe_graph():
    graph = StateGraph(PocState)
    _add_dual_swe_nodes(graph)
    _wire_entry(graph, "dual-swe")
    _wire_execution(graph, "dual-swe")
    _wire_review(graph, "dual-swe")
    return graph.compile()


def build_loop_graph():
    graph = StateGraph(PocState)
    _add_loop_nodes(graph)
    _wire_entry(graph, "loop")
    _wire_execution(graph, "loop")
    return graph.compile()


def build_graph(mode: str = "single"):
    if mode == "single":
        return build_single_graph()
    if mode == "multi":
        return build_multi_graph()
    if mode == "negotiate":
        return build_negotiate_graph()
    if mode == "dual-swe":
        return build_dual_swe_graph()
    if mode == "loop":
        return build_loop_graph()
    raise ValueError(f"Unsupported graph mode: {mode}")
