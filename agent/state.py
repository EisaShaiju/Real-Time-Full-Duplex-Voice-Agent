"""The turn state machine.

    LISTENING  --speech, then endpoint-->  THINKING
    THINKING   --first audio out-------->  SPEAKING
    SPEAKING   --user speaks------------>  INTERRUPTED --> LISTENING
    THINKING   --user speaks------------>  INTERRUPTED --> LISTENING

Every transition is logged with how long we spent in the previous state. This
single log is where interrupt bugs surface, so it's kept deliberately verbose.
"""

import logging
import time
from enum import Enum

log = logging.getLogger("state")


class State(Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class StateMachine:
    def __init__(self):
        self.state = State.LISTENING
        self._entered = time.perf_counter()

    def to(self, new: State):
        if new == self.state:
            return
        now = time.perf_counter()
        log.info(
            "%-11s -> %-11s (%.0f ms in %s)",
            self.state.value, new.value, (now - self._entered) * 1000, self.state.value,
        )
        self.state = new
        self._entered = now

    def is_(self, state: State) -> bool:
        return self.state == state
