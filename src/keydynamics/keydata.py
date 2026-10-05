from dataclasses import dataclass
from typing import Literal


@dataclass()
class KeyData:
    participant_id: int
    session_id: int
    sample_id: int
    event: Literal["PRESS", "RELEASE"]
    key_code: int
    timestamp: float
    press_id: int
    key: str

    def __init__(self, participant_id, session_id, sample_id, event, key_code, timestamp, press_id, key):
        self.participant_id = participant_id
        self.session_id = session_id
        self.sample_id = sample_id
        self.event = event
        self.key_code = key_code
        self.timestamp = timestamp
        self.press_id = press_id
        self.key = key

