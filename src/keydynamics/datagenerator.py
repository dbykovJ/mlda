from dataclasses import dataclass

from keydynamics.keydata import KeyData


@dataclass()
class DataGenerator:
    participant_id: int
    session_id: int
    sample_id: int
    press_id: int

    def __init__(self, participant_id, session_id):
        self.participant_id = participant_id
        self.session_id = session_id
        self.sample_id = 0
        self.press_id = 0


    def registerPress(self, key_code, timestamp, key):
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "UP", key_code, timestamp, self.press_id, key)
        self.press_id += 1

    def registerRelease(self, key_code, timestamp, key):
        keyData = KeyData(self.participant_id, self.session_id, self.sample_id, "DOWN", key_code, timestamp, self.press_id, key)







