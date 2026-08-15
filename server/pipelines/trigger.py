"""Confusion phrase detector → fires Mode 2."""

import re
import time

from server.config import TRIGGER_COOLDOWN_S

CONFUSION_PATTERNS = [
    r"i don'?t remember",
    r"who (are|is) (you|she|he|they)",
    r"what did (we|you) (talk|say|discuss)",
    r"i (forget|forgot)",
    r"remind me",
    r"tell me about",
    r"who was that",
    r"i can'?t recall",
    r"what'?s (your|her|his) name",
    r"where (am|are) i",
    r"what (year|day|month) is it",
]


class TriggerDetector:
    """
    Regex-based detector on STT output.
    Fast — no ML inference. Cooldown prevents spam.
    """

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in CONFUSION_PATTERNS]
        self._last_triggered: float = -TRIGGER_COOLDOWN_S  # allow first trigger

    def check(self, text: str, current_time: float | None = None) -> bool:
        if current_time is None:
            current_time = time.time()
        if current_time - self._last_triggered < TRIGGER_COOLDOWN_S:
            return False
        for pattern in self.patterns:
            if pattern.search(text):
                self._last_triggered = current_time
                return True
        return False
