# LangGraph + mini-SWE-agent + LiteLLM PoC

This is a small local proof of concept:

```text
LangGraph coordinator
  -> prepare task
  -> run mini-SWE-agent worker in Docker
  -> review/collect result

mini-SWE-agent
  -> LiteLLM model
  -> Docker sandbox
```

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp .env.example .env
```

The default is local Ollama:

```text
MSWEA_MODEL_NAME=qwen3:32b
LITELLM_API_BASE=http://localhost:11436
MSWEA_COST_TRACKING=ignore_errors
MSWEA_TRAJECTORY_PATH=trajectories/last_run.traj.json
```

The worker sends this to LiteLLM as `ollama/qwen3:32b`.

## Smoke Test Without LLM/Docker

```bash
uv run langgraph-mini-swe --mock "Create a hello.py script"
```

Every CLI run saves the full final LangGraph state as JSON. By default, the
state path is derived from `--output`:

```text
trajectories/example.traj.json  ->  trajectories/example.state.json
```

You can override it:

```bash
uv run langgraph-mini-swe \
  --mock \
  --mode negotiate \
  --output trajectories/negotiate-debug.traj.json \
  --state-output runs/negotiate-debug.state.json \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

Inspect useful fields:

```bash
jq '.conversation' trajectories/negotiate-debug.state.json
jq '.worker_task' trajectories/negotiate-debug.state.json
jq '.final' trajectories/negotiate-debug.state.json
```

## Visualize The Graph

```bash
uv run langgraph-mini-swe --graph
uv run langgraph-mini-swe --mode multi --graph
uv run langgraph-mini-swe --mode negotiate --graph
uv run langgraph-mini-swe --mode dual-swe --graph
```

Single-agent graph:

```mermaid
flowchart TD
    Start([START]) --> Prepare[prepare_task]
    Prepare --> Worker[run_worker]
    Worker --> Review[review_result]
    Review --> End([END])
```

Multi-agent graph:

```mermaid
flowchart TD
    Start([START]) --> Planner[planner_agent]
    Planner --> Prepare[prepare_task]
    Prepare --> Worker[run_worker]
    Worker --> Reviewer[reviewer_agent]
    Reviewer --> Review[review_result]
    Review --> End([END])
```

Negotiation graph:

```mermaid
flowchart TD
    Start([START]) --> Planner[planner_agent]
    Planner --> PlanReviewer[plan_reviewer_agent]
    PlanReviewer --> Revise[revise_plan_agent]
    Revise --> Prepare[prepare_task]
    Prepare --> Worker[run_worker]
    Worker --> Reviewer[reviewer_agent]
    Reviewer --> Review[review_result]
    Review --> End([END])
```

Dual-SWE graph:

```mermaid
flowchart TD
    Start([START]) --> Planner[planner_agent]
    Planner --> Prepare[prepare_task]
    Prepare --> Implementer[run_implementer_swe]
    Implementer --> VerifierTask[prepare_verifier_task]
    VerifierTask --> Verifier[run_verifier_swe]
    Verifier --> Reviewer[reviewer_agent]
    Reviewer --> Review[review_result]
    Review --> End([END])
```

## Run LangGraph Studio

This project includes `langgraph.json`, so LangGraph Studio can load:

- `mini_swe_poc`
- `mini_swe_multi`
- `mini_swe_negotiate`
- `mini_swe_dual`
- `mini_swe_loop`

```bash
uv run langgraph dev --no-browser --no-reload
```

Make sure `.env` contains a LangSmith key before starting Studio:

```bash
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=false
```

`langgraph.json` is configured with `"env": ".env"`, so restart
`langgraph dev` after changing `.env`.

Open the Studio URL printed by the command:

```text
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

The local API is available at:

```text
http://127.0.0.1:2024
```

Use this mock input first if you only want to inspect graph state without
calling an LLM or starting Docker:

```json
{
  "task": "Create /tmp/hello.py that prints hello, run it, then submit the result",
  "mode": "negotiate",
  "model": "openai/gpt-4.1",
  "image": "python:3.11",
  "cwd": "/tmp",
  "step_limit": 20,
  "cost_limit": 1.0,
  "cost_tracking": "ignore_errors",
  "timeout": 60,
  "api_base": null,
  "output_path": "trajectories/studio-negotiate.traj.json",
  "shared_dir": "",
  "shared_mount": "/workspace/shared",
  "mock": true
}
```

Useful state fields to inspect:

- `planner_message`
- `plan_review_message`
- `revised_planner_message`
- `conversation`
- `worker_task`
- `worker_result`
- `reviewer_message`
- `final`

If Studio submits an empty/default payload, the graph now fills a default
`task`. You can also use a chat-style input:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Create /tmp/hello.py that prints hello, run it, then submit the result"
    }
  ],
  "mock": true
}
```

## Run With mini-SWE-agent + Docker

Make sure Docker Desktop is running.

```bash
uv run langgraph-mini-swe \
  --model qwen2.5-coder:32b-instruct-q4_K_M \
  --api-base http://localhost:11436 \
  --output trajectories/qwen-hello.traj.json \
  --image python:3.11 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

or

```bash
uv run langgraph-mini-swe \
  --model openai/gpt-4.1 \
  --output trajectories/openai-hello.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

or

```bash
uv run langgraph-mini-swe \
  --model anthropic/claude-sonnet-4-5-20250929 \
  --output trajectories/claude-hello.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```


The trajectory is saved to the `--output` path. By default this is
`trajectories/last_run.traj.json`.

## Run Multi-Agent Mode

This adds a Planner agent before the SWE worker and a Reviewer agent after it.
They communicate through LangGraph state in the `conversation` field.

```bash
export OPENAI_API_KEY="..."

uv run langgraph-mini-swe \
  --mode multi \
  --model openai/gpt-4.1 \
  --output trajectories/multi-openai-hello.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

## Run Coordinator Loop Mode

This mode keeps multiple isolated agents, runs them in parallel each round, and
lets the coordinator repeat for N rounds. Each agent keeps its own history.

Each agent runs in its own Docker container concurrently (one container per
agent per round).

Default round plan:

- Round 1: implement Fibonacci (recursive, matrix, fast doubling)
- Round 2: add tests for each implementation
- Round 3: run with input and show execution logs

The main CLI task string is appended as a coordinator narrative for every agent.

Note: agent code and test files live inside the Docker sandbox by default.
Use a shared mount if you want outputs on the host.

```bash
export OPENAI_API_KEY="..."

uv run langgraph-mini-swe \
  --mode loop \
  --rounds 3 \
  --agents agent_a,agent_b,agent_c \
  --runtime-input 10 \
  --model openai/gpt-4.1 \
  --output trajectories/loop-openai-fib.traj.json \
  --image python:3.11 \
  "Implement Fibonacci with isolated agents"
```

Example with a shared output directory:

```bash
export OPENAI_API_KEY="..."

uv run langgraph-mini-swe \
  --mode loop \
  --rounds 3 \
  --agents agent_a,agent_b,agent_c \
  --runtime-input 10 \
  --model openai/gpt-4.1 \
  --output trajectories/loop-openai-fib.traj.json \
  --image python:3.11 \
  --shared-dir ./runs/shared \
  --shared-mount /workspace/shared \
  "Implement Fibonacci with isolated agents. Write all outputs to /workspace/shared."
```

## Run Negotiation Mode

This mode lets Planner and Plan Reviewer negotiate before the SWE worker runs:

```text
planner_agent -> plan_reviewer_agent -> revise_plan_agent -> run_worker -> reviewer_agent
```

```bash
export OPENAI_API_KEY="..."

uv run langgraph-mini-swe \
  --mode negotiate \
  --model openai/gpt-4.1 \
  --output trajectories/negotiate-openai-hello.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

## Run Two SWE Agents

This runs two separate mini-SWE-agent workers:

1. `run_implementer_swe` works on the task in one sandbox.
2. `collect_implementer_artifact` parses the implementer's submitted artifact
   content string.
3. `run_verifier_swe` receives that string through LangGraph state, recreates
   the file in a fresh sandbox, and verifies it.

Each SWE worker gets its own trajectory:

- `trajectories/dual-openai-hello.implementer.traj.json`
- `trajectories/dual-openai-hello.verifier.traj.json`

```bash
export OPENAI_API_KEY="..."

uv run langgraph-mini-swe \
  --mode dual-swe \
  --model openai/gpt-4.1 \
  --output trajectories/dual-ip-handoff.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Agent A calls curl -4 icanhazip.com, writes a python file named print_ip.py whose content prints that IP. Verifier must run the file, call curl https://api.ipify.org, and pass only if both IP values match."
```

The default dual-SWE handoff does not use a shared filesystem. Agent A submits:

```text
ARTIFACT_PATH=print_ip.py
ARTIFACT_CONTENT_BASE64=...
```

LangGraph decodes that into `artifact_report.artifact_content` and passes it
to Agent B.

## Use Another LiteLLM Endpoint

Point mini-SWE-agent at another LiteLLM-compatible endpoint:

```bash
uv run langgraph-mini-swe \
  --model openai/my-routed-model \
  --api-base http://localhost:4000/v1 \
  "Write and run a small Python factorial script"
```

## Try OpenAI

Set your OpenAI key:

```bash
export OPENAI_API_KEY="..."
```

Then omit `--api-base` and use an OpenAI model:

```bash
uv run langgraph-mini-swe \
  --model openai/gpt-4.1 \
  --output trajectories/openai-hello.traj.json \
  --image python:3.11 \
  --step-limit 20 \
  "Create /tmp/hello.py that prints hello, run it, then submit the result"
```

You can also pass `--model gpt-4.1`; the wrapper normalizes it to
`openai/gpt-4.1`.

## Files

- `src/langgraph_mini_swe_poc/main.py`: CLI entrypoint.
- `src/langgraph_mini_swe_poc/graph.py`: LangGraph orchestration.
- `src/langgraph_mini_swe_poc/worker.py`: mini-SWE-agent worker wrapper.

The important design choice is that LangGraph does not know much about mini-SWE-agent. It just calls `run_swe_worker(...)`, which you can later replace with SWE-ReX, AIO Sandbox, Modal, or multiple parallel workers.
