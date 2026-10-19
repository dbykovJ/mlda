from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.keydynamics.keydata import KeyData


@dataclass
class SampleProps:
    dwell_time: float
    press_press: float
    press_release: float
    release_press: float
    release_release: float


    def from_sample(cls, sample: list[KeyData]) -> "SampleProps | None":
        df = pd.DataFrame([asdict(k) for k in sample])
 
        press = df[df.event == "PRESS"]
        release = df[df.event == "RELEASE"]
        keys = press.merge(
            release[["press_id", "timestamp"]],
            on="press_id",
            suffixes=("_p", "_r"),
        ).rename(columns={"timestamp_p": "p", "timestamp_r": "r"})
        keys = keys.sort_values("p")
 
        if len(keys) < 2:
            return None  
 
        H = keys.r - keys.p                  # r_i - p_i
        DD = keys.p.shift(-1) - keys.p       # p_{i+1} - p_i
        UD = keys.p.shift(-1) - keys.r       # p_{i+1} - r_i
        UU = keys.r.shift(-1) - keys.r       # r_{i+1} - r_i
        DU = keys.r.shift(-1) - keys.p       # r_{i+1} - p_i
 
        return cls(
            dwell_time=H.mean(),
            press_press=DD.mean(),
            press_release=DU.mean(),
            release_press=UD.mean(),
            release_release=UU.mean(),
        )
 
    def to_np(self) -> np.ndarray:
        return np.array(
            [
                self.dwell_time,
                self.press_press,
                self.press_release,
                self.release_press,
                self.release_release,
            ]
        )

        
    

@dataclass
class KeyModel:
    sample_props: list[SampleProps]

    def from_csv(cls, path: str) -> "KeyModel":
        df = pd.read_csv(path)
        props = []
        for _, group in df.groupby(["session_id", "sample_id"]):
            sample = [KeyData(**row) for row in group.to_dict("records")]
            p = SampleProps.from_sample(sample)
            if p is not None:
                props.append(p)
        return cls(sample_props=props)
 
    def save(self, path: str):
        np.save(path, self.to_np())
 
    def to_np(self) -> np.ndarray:
        return np.array([sample.to_np() for sample in self.sample_props])
 
    @classmethod
    def from_file(cls, path: str) -> "KeyModel":
        return cls(sample_props=[SampleProps(*row) for row in np.load(path)])
