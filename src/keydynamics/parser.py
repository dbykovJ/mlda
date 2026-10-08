import argparse

def parserSetup(parser: argparse.ArgumentParser):
    parser.add_argument("participant_id", type=int)
    parser.add_argument("output_path", type=str)
    parser.add_argument("session_timeout", type=float)

    return parser