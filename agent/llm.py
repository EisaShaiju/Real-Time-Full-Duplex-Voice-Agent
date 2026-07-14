"""Streaming LLM client on Groq.

``stream(text)`` is an async generator that yields token strings as they arrive.
Cancelling the task that drives it (what barge-in does) stops generation; the
``finally`` block still records whatever partial reply was produced so the
conversation history stays coherent for the next turn.
"""

import os

from groq import AsyncGroq

from . import config


class LLM:
    def __init__(self, model=config.LLM_MODEL, system_prompt=config.SYSTEM_PROMPT):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GROQ_API_KEY environment variable.")
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.history = [{"role": "system", "content": system_prompt}]

    async def stream(self, user_text):
        """Yield tokens for the reply to ``user_text``. Cancel to stop early."""
        self.history.append({"role": "user", "content": user_text})
        produced = []
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            temperature=config.LLM_TEMPERATURE,
        )
        try:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    produced.append(token)
                    yield token
        finally:
            # Save even a partial reply so history reflects what the user heard.
            self.history.append({"role": "assistant", "content": "".join(produced)})
