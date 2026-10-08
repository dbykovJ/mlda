import csv
from dataclasses import asdict, fields
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
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
