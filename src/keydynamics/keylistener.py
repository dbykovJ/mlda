import os
import sys
import time
from pathlib import Path

from keydynamics.datagenerator import DataGenerator

if sys.platform == "linux":
    os.environ["PYNPUT_BACKEND_KEYBOARD"] = "uinput"
    os.environ["PYNPUT_BACKEND_MOUSE"] = "dummy"
    helper_dir = Path(__file__).resolve().parents[2] / "scripts" / "uinput-bin"
    os.environ["PATH"] = str(helper_dir) + os.pathsep + os.environ.get("PATH", "")

from pynput.keyboard import Key, KeyCode, Listener


def select_linux_device() -> str:
    import evdev
    from evdev import ecodes

    keyboards: list[tuple[str, str]] = []

    i = 0

    print("Available keyboards: Please select")

    for path in evdev.list_devices():  # pyright: ignore[reportUnknownMemberType]
        device = evdev.InputDevice(path, readonly=True)  # pyright: ignore[reportUnknownVariableType]
        try:
            keys = device.capabilities().get(ecodes.EV_KEY, [])
            if ecodes.KEY_A in keys and ecodes.KEY_ENTER in keys:
                keyboards.append((device.name, str(device.path)))

                print(f"{i + 1}) {device.name}")
                i = i + 1
        finally:
            device.close()

    if not keyboards:
        sys.exit("No accessible keyboards found.")

    try:
        idx = int(input(f"Select 1-{i}: "))
        if not 1 <= idx <= len(keyboards):
            raise ValueError
    except ValueError:
        sys.exit("Not a valid selection, exiting...")

    name, path = keyboards[idx - 1]
    print(f"selected device: {name}")
    return path


class KeyListener:
    def __init__(self, dataGenerator: DataGenerator):
        self.dataGenerator: DataGenerator = dataGenerator
        options = {}
        if sys.platform == "linux":
            options["uinput_device_paths"] = [select_linux_device()]
        self.listener: Listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release,
            **options,  # pyright: ignore[reportUnknownArgumentType]
        )

    def on_press(self, key: Key | KeyCode | None) -> None:
        if key is None or key == Key.esc:
            return
        code = key_code(key)
        if code is None:
            return

        self.dataGenerator.registerPress(
            key_code=code, timestamp=time.time(), key=key_char(key)
        )

    def on_release(self, key: Key | KeyCode | None) -> None | bool:
        if key is None:
            return None
        if key == Key.esc:
            return False
        code = key_code(key)
        if code is None:
            return None
        self.dataGenerator.registerRelease(
            key_code=code, timestamp=time.time(), key=key_char(key)
        )
        return None

    def listen(self) -> None:
        print("Listening to keys...")
        self.listener.start()
        try:
            self.listener.join()
        finally:
            self.listener.stop()


# wrappers around the keyboard types so the IDE doesn't complain
def key_code(key: Key | KeyCode) -> int | None:
    if isinstance(key, Key):
        return key.value.vk
    return key.vk


def key_char(key: Key | KeyCode) -> str | None:
    if isinstance(key, Key):
        return key.name  # some keys don't have chars (shift etc.)
    return key.char
