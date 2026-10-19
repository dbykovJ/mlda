from dataclasses import dataclass, field
from queue import Queue
from uuid import UUID

from keydynamics.keydata import KeyData


@dataclass()
class DataGenerator:
    participant_id: int
    session_id: UUID
    memory: Queue[KeyData]
    sample_timeout: float = 5.0
    pressed_keys: dict[int, int] = field(default_factory=dict)
    sample_id: int = 0
    press_id: int = 0
    last_event_time: float | None = None

    def registerPress(self, key_code: int, timestamp: float, key: str | None) -> None:
        if key_code in self.pressed_keys:
            return
        self.check_new_session(timestamp)
        keyData = KeyData(
            self.participant_id,
            self.session_id,
            self.sample_id,
            "PRESS",
            key_code,
            timestamp,
            self.press_id,
            key,
        )
        self.pressed_keys[key_code] = self.press_id
        self.press_id += 1
        self.memory.put(keyData)

    def registerRelease(self, key_code: int, timestamp: float, key: str | None) -> None:
        press_id = self.pressed_keys.get(key_code)
        if press_id is None:
            # A key may have been pressed before the listener started.
            return
        self.last_event_time = timestamp
        keyData = KeyData(
            self.participant_id,
            self.session_id,
            self.sample_id,
            "RELEASE",
            key_code,
            timestamp,
            press_id,
            key,
        )
        _ = self.pressed_keys.pop(key_code)
        self.memory.put(keyData)

    def check_new_session(self, timestamp: float) -> None:
        if (
            self.last_event_time is not None
            and not self.pressed_keys
            and timestamp - self.last_event_time > self.sample_timeout
        ):
            self.sample_id += 1
        self.last_event_time = timestamp
