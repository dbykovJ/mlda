from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue

from keydynamics.keydata import KeyData


@dataclass()
class DataGenerator:
    participant_id: int
    session_id: int
    memory: Queue
    pressed_keys: dict[str, int] = field(default_factory=dict)
    sample_id: int = 0
    press_id: int = 0


    def registerPress(self, key_code, timestamp, key):
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "PRESS", key_code, timestamp, self.press_id, key)
        if key_code in self.pressed_keys:
            raise KeyError("Key already registered")
        self.pressed_keys[key_code] = self.press_id
        self.press_id += 1
        self.memory.put(keyData)

    def registerRelease(self, key_code, timestamp, key):
        press_id = self.pressed_keys[key_code]
        if press_id is None:
            raise ValueError("Release key for an unpressed key registered")
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "RELEASE", key_code, timestamp, press_id, key)
        self.pressed_keys.pop(key_code)
        self.memory.put(keyData)


    def incSampleId(self):
        self.sample_id += 1





