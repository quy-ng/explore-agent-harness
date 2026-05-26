from __future__ import annotations

from typing import Any

from langgraph_mini_swe_poc.worker import normalize_litellm_model


def chat_once(
    *,
    role: str,
    prompt: str,
    model: str,
    api_base: str | None,
    mock: bool = False,
) -> dict[str, Any]:
    litellm_model = normalize_litellm_model(model, api_base)
    if mock:
        return {
            "role": role,
            "content": f"[mock {role}] {prompt[:240]}",
            "model": model,
            "litellm_model": litellm_model,
            "api_base": api_base,
        }

    from litellm import completion

    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are the {role} in a small multi-agent coding workflow. "
                    "Be concise, concrete, and useful to the next agent."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "drop_params": True,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = completion(**kwargs)
    content = response.choices[0].message.content or ""
    return {
        "role": role,
        "content": content.strip(),
        "model": model,
        "litellm_model": litellm_model,
        "api_base": api_base,
    }
