import argparse
import glob
import os
import uuid
from queue import Queue

from keydynamics.csvwriter import PeriodicWrite, save_queue
from keydynamics.datagenerator import DataGenerator
from keydynamics.parser import parserSetup


def get_session_id(output_path: str) -> int:
    list_of_files = glob.glob(output_path + "/*.csv")
    if not list_of_files:
        latest_file = output_path + "/keydynamics.csv"
    else:
        latest_file = max(list_of_files, key=os.path.getctime)

    return hash(latest_file)


def main() -> None:
    from keydynamics.keylistener import KeyListener

    print("Hello from keydynamics!")

    parser = argparse.ArgumentParser(prog="keydynamics")
    parser = parserSetup(parser)

    args = parser.parse_args()
    out_path = args.output_path

    session_id = uuid.uuid4()

    memory = Queue()

    dataGenerator = DataGenerator(
        session_id=session_id,
        participant_id=args.participant_id,
        memory=memory,
        sample_timeout=args.sample_timeout,
    )
    keyListener = KeyListener(dataGenerator)

    periodicWrite = PeriodicWrite(5, save_queue, memory=memory, path=out_path)
    periodicWrite.start()

    try:
        keyListener.listen()
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        periodicWrite.stop()
        periodicWrite.join()
        save_queue(memory, out_path)


if __name__ == "__main__":
    main()
