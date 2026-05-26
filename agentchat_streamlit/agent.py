import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient


class Agent:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        load_dotenv(repo_root / ".env")

        # Load the model client from config.
        config_path = Path(__file__).resolve().parent / "model_config.yml"
        with config_path.open("r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)

        if "api_key" not in model_config.get("config", {}):
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                model_config.setdefault("config", {})["api_key"] = api_key

        model_client = ChatCompletionClient.load_component(model_config)
        self.agent = AssistantAgent(
            name="assistant",
            model_client=model_client,
            system_message="You are a helpful AI assistant.",
        )

    async def chat(self, prompt: str) -> str:
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")],
            CancellationToken(),
        )
        assert isinstance(response.chat_message, TextMessage)
        return response.chat_message.content