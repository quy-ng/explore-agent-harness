from __future__ import annotations

import asyncio
from pathlib import Path
import platform
from typing import Any

import yaml


def normalize_litellm_model(model: str, api_base: str | None) -> str:
    if "/" in model:
        return model
    if api_base and "11436" in api_base:
        return f"ollama/{model}"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{model}"
    return model


def run_swe_worker(
    *,
    task: str,
    model: str,
    image: str,
    cwd: str,
    step_limit: int,
    cost_limit: float,
    cost_tracking: str,
    timeout: int,
    api_base: str | None = None,
    output_path: str | None = None,
    docker_args: list[str] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    litellm_model_name = normalize_litellm_model(model, api_base)
    if mock:
        return {
            "exit_status": "Mocked",
            "submission": f"Would run mini-SWE-agent on: {task}",
            "model": model,
            "litellm_model": litellm_model_name,
            "api_base": api_base,
            "trajectory_path": output_path,
            "docker_args": docker_args or [],
            "image": image,
            "cwd": cwd,
        }

    from minisweagent import package_dir
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    config_path = Path(package_dir) / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())

    agent_config = dict(config.get("agent", {}))
    agent_config["step_limit"] = step_limit
    agent_config["cost_limit"] = cost_limit
    if output_path:
        trajectory_path = Path(output_path)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        agent_config["output_path"] = trajectory_path

    model_config = dict(config.get("model", {}))
    model_kwargs: dict[str, Any] = dict(model_config.get("model_kwargs", {}))
    model_kwargs["drop_params"] = True
    model_kwargs["tool_choice"] = "required"
    if api_base:
        model_kwargs["api_base"] = api_base

    litellm_model = LitellmModel(
        model_name=litellm_model_name,
        model_kwargs=model_kwargs,
        cost_tracking=cost_tracking,
        format_error_template=model_config.get("format_error_template", "{{ error }}"),
        observation_template=model_config.get("observation_template"),
    )
    environment = SwerexDockerEnvironment(
        image=image,
        cwd=cwd,
        timeout=timeout,
        deployment_extra_kwargs={"docker_args": docker_args or []},
    )
    agent = DefaultAgent(litellm_model, environment, **agent_config)
    system_info = platform.uname()._asdict()

    try:
        result = agent.run(task, **system_info)
    finally:
        asyncio.run(environment.deployment.stop())

    return {
        "exit_status": result.get("exit_status", ""),
        "submission": result.get("submission", ""),
        "model": model,
        "litellm_model": litellm_model_name,
        "api_base": api_base,
        "trajectory_path": str(agent_config.get("output_path", "")),
        "docker_args": docker_args or [],
        "image": image,
        "cwd": cwd,
        "raw": result,
    }
