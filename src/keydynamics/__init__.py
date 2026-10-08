from queue import Queue
import argparse
import os
import glob

from keydynamics.keylistener import KeyListener
from keydynamics.datagenerator import DataGenerator

from keydynamics.parser import parserSetup

def get_session_id(output_path: str):
    list_of_files = glob.glob(output_path + '/*.csv')
    if not list_of_files:
        latest_file = output_path + "/keydynamics.csv"
    else:
        latest_file = max(list_of_files, key=os.path.getctime)

    return hash(latest_file)

def main() -> None:
    print("Hello from keydynamics!")

    parser = argparse.ArgumentParser(prog="keydynamics")
    parser = parserSetup(parser)

    args = parser.parse_args()

    session_id = get_session_id(args.output_path)

    memory = Queue()
    dataGenerator = DataGenerator(session_id=session_id, participant_id=args.participant_id, memory=memory)
    keyListener = KeyListener(dataGenerator)
    keyListener.listen()


if __name__ == "__main__":
    main()