"""Run the live full-duplex voice agent.

    python main.py

Speak, and the agent replies. Talk over it and it stops, listens, and responds
to the new input. Ctrl+C to quit. Use headphones so the mic doesn't hear the
agent's own voice.
"""

import asyncio
import logging

from agent.pipeline import Pipeline


async def amain():
    agent = Pipeline()
    print("Listening. Talk to me, and feel free to interrupt. (Ctrl+C to quit)\n")
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nbye")
