import csv
from dataclasses import asdict, fields
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from keydynamics.keydata import KeyData


def save_queue(memory: Queue[KeyData], path: str) -> None:
    rows: list[KeyData] = []
    while True:
        try:
            rows.append(memory.get_nowait())
        except Empty:
            break

    fieldnames = [fld.name for fld in fields(KeyData)]
    _path = Path(path)

    _path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not _path.exists() or _path.stat().st_size == 0

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    if rows:
        print(f"Written {len(rows)} events to {path}")


class PeriodicWrite(Thread):
    def __init__(self, interval, function, **kwargs):
        super().__init__(daemon=True)
        self.interval = interval
        self.function = function
        self.kwargs = kwargs
        self._stop_event = Event()

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.function(**self.kwargs)

    def stop(self):
        self._stop_event.set()
