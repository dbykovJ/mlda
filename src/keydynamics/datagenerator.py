from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue

from keydynamics.keydata import KeyData


@dataclass()
class DataGenerator:
    participant_id: int
    session_id: int
    memory: Queue
    session_timeout: float = 5.0
    pressed_keys: dict[str, int] = field(default_factory=dict)
    sample_id: int = 0
    press_id: int = 0
    last_event_time: float | None = None


    def registerPress(self, key_code, timestamp, key) -> None:
        if key_code in self.pressed_keys:
            return
        self.check_new_session(timestamp)
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "PRESS", key_code, timestamp, self.press_id, key)
        self.pressed_keys[key_code] = self.press_id
        self.press_id += 1
        self.memory.put(keyData)

    def registerRelease(self, key_code, timestamp, key) -> None:
        press_id = self.pressed_keys[key_code]
        if press_id is None:
            raise ValueError("Release key for an unpressed key registered")
        self.last_event_time = timestamp
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "RELEASE", key_code, timestamp, press_id, key)
        self.pressed_keys.pop(key_code)
        self.memory.put(keyData)
        
    def check_new_session(self, timestamp) -> None:
        if self.last_event_time is not None and timestamp - self.last_event_time > self.session_timeout:
            self.sample_id +=1
        self.last_event_time = timestamp





