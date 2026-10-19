from dataclasses import dataclass
from typing import Literal
from uuid import UUID


@dataclass()
class KeyData:
    participant_id: int
    session_id: UUID
    sample_id: int
    event: Literal["PRESS", "RELEASE"]
    key_code: int
    timestamp: float
    press_id: int
    key: str | None
