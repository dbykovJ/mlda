from dataclasses import dataclass
from typing import Literal


@dataclass()
class Keydata:
    participant_id: int
    session_id: int
    sample_id: int
    event: Literal["PRESS", "RELEASE"]
    key_code: int
    timestamp: float
    press_id: int
    key: str
