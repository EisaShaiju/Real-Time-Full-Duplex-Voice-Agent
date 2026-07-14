"""Semantic endpointing -- the clever bit.

Naive VAD ends a turn on *any* pause, so it cuts you off when you stop to think
mid-sentence ("Book me a table for... uh... four people"). Here we look at the
partial transcript when VAD detects a pause and ask: is this a complete thought?

Two stages, cheapest first:

1. A fast heuristic gives a confident verdict for the easy cases (ends in
   punctuation -> done; ends on a filler / too short -> keep listening).
2. Only genuinely ambiguous cases pay for a 1-token Groq call.

This mirrors how production voice stacks keep endpointing latency near zero for
the common case while still handling the hard ones.
"""

import os

from groq import AsyncGroq

from . import config

# Words that almost never end a finished sentence -> the user is mid-thought.
_TRAILING_INCOMPLETE = {
    "um", "uh", "er", "and", "but", "or", "so", "because", "the", "a", "an",
    "to", "of", "for", "with", "my", "your", "i", "we", "that", "if", "when",
}


def quick_verdict(text: str):
    """Return True (complete), False (incomplete), or None (ask the LLM)."""
    t = text.strip()
    if not t:
        return False
    if t[-1] in ".?!":
        return True
    words = t.lower().split()
    if len(words) < 2:
        return False
    if words[-1].strip(",;:") in _TRAILING_INCOMPLETE:
        return False
    return None  # ambiguous -> escalate


class SemanticEndpointer:
    def __init__(self, use_llm=config.USE_SEMANTIC_ENDPOINTING, model=config.ENDPOINT_LLM_MODEL):
        self.use_llm = use_llm
        self.model = model
        self._client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"]) if use_llm else None

    async def is_complete(self, text: str) -> bool:
        verdict = quick_verdict(text)
        if verdict is not None:
            return verdict
        if not self.use_llm:
            return True  # bias toward responding rather than hanging
        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=1,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide if the user's utterance is a complete thought they "
                        "expect a reply to. Answer with exactly 1 (complete) or 0 "
                        "(they seem mid-sentence)."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip().startswith("1")
