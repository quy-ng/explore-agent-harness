from typing import List, Dict
from pathlib import Path
from litellm import completion


class MemoryManager:
    """Simple sliding-window memory with summarization for chat.

    - Keeps recent `window_size` messages in `history`.
    - When history grows beyond window_size, summarizes the oldest chunk
      and appends to `summary` string, then deletes those messages.
    - `get_context()` returns a system summary message (if present)
      followed by the recent history.
    """

    def __init__(self, model_config: Dict = None, window_size: int = 8) -> None:
        self.history: List[Dict[str, str]] = []
        self.summary: str = ""
        self.window_size = window_size
        self.model_config = model_config or {}

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        # If history too long, summarize oldest messages
        if len(self.history) > self.window_size:
            overflow = len(self.history) - self.window_size
            to_summarize = self.history[:overflow]
            # Build summarization prompt
            convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
            prompt = (
                "Summarize the following conversation briefly (2-3 sentences), "
                "keeping important context and decisions.\n\n" + convo_text
            )
            # Use liteLLM completion synchronously
            try:
                resp = completion(
                    messages=[
                        {"role": "system", "content": "You are a concise summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_config,
                )
                summary_text = resp.choices[0].message.content or ""
            except Exception:
                summary_text = ""

            if summary_text:
                if self.summary:
                    self.summary += "\n" + summary_text
                else:
                    self.summary = summary_text

            # Drop the summarized messages from history
            self.history = self.history[overflow:]

    def get_context(self) -> List[Dict[str, str]]:
        context: List[Dict[str, str]] = []
        if self.summary:
            context.append({"role": "system", "content": f"Conversation summary:\n{self.summary}"})
        context.extend(self.history)
        return context
