from pynput import keyboard

from keydynamics.keylistener import Keylistener


def main() -> None:
    print("Hello from keydynamics!")
    print("listeing to keys...")
    keyListener = Keylistener()
    keyListener.listen()