import argparse

def parserSetup(parser: argparse.ArgumentParser):
    parser.add_argument("participant_id", type=int)
    parser.add_argument("output_path", type=str)
    parser.add_argument("idle_interval", type=float)

    return parser