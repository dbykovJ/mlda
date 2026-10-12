import csv
import threading
from dataclasses import asdict, fields
from pathlib import Path
from queue import Empty, Queue

from keydynamics.keydata import KeyData


def save_queue(memory: Queue, path: str) -> None:
    rows = []
    while True:
        try:
            rows.append(memory.get_nowait())
        except Empty:
            break

    fieldnames = [fld.name for fld in fields(KeyData)]
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print("Written")


class PeriodicWrite(threading.Thread):
    def __init__(self, interval, function, **kwargs):
        super().__init__(daemon=True)
        self.interval = interval
        self.function = function
        self.kwargs = kwargs
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.function(**self.kwargs)

    def stop(self):
        self._stop_event.set()