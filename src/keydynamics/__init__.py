from queue import Queue

from pynput import keyboard

from keydynamics.keylistener import KeyListener
from keydynamics.datagenerator import DataGenerator

def main() -> None:
    print("Hello from keydynamics!")
    memory = Queue()
    dataGenerator = DataGenerator(1, 1, memory)
    keyListener = KeyListener(dataGenerator)
    keyListener.listen()


if __name__ == "__main__":
    main()