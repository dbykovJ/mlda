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