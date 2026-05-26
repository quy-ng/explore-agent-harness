import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from litellm import completion
from .memory import MemoryManager
from .cache import LRUCache
import json
import hashlib


class Agent:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        load_dotenv(repo_root / ".env")

        config_path = Path(__file__).resolve().parent / "model_config.yml"
        with config_path.open("r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f) or {}

        self.model_config = dict(model_config)
        self.model_config.setdefault("model", "gpt-4o-mini")
        # sliding-window memory manager
        self.memory = MemoryManager(model_config=self.model_config, window_size=8)
        # in-process LRU cache for responses
        self.cache = LRUCache(max_size=256)

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and "api_key" not in self.model_config:
            self.model_config["api_key"] = api_key

        api_base = os.getenv("LITELLM_API_BASE")
        if api_base and "api_base" not in self.model_config:
            self.model_config["api_base"] = api_base

    def chat(self, prompt: str) -> str:
        # record user message
        self.memory.add_message("user", prompt)

        # build context: summary + recent history
        messages = self.memory.get_context()
        # add the current user message at the end if not present
        if not messages or messages[-1].get("content") != prompt:
            messages = messages + [{"role": "user", "content": prompt}]

        # compute cache key from context + model config
        key_payload = {"messages": messages, "model_config": self.model_config}
        key_raw = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

        cached = self.cache.get(key)
        if cached is not None:
            content = cached
        else:
            response = completion(messages=messages, **self.model_config)
            content = response.choices[0].message.content or ""
            # store result in cache
            try:
                self.cache.set(key, str(content))
            except Exception:
                pass

        # record assistant reply
        self.memory.add_message("assistant", content)
        return str(content)
