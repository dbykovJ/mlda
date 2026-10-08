import time

from pynput import keyboard

# wrappers around the keyboard types so the IDE doesn't complain
def key_code(key: keyboard.Key | keyboard.KeyCode) -> int | None:
    if isinstance(key, keyboard.Key):
        return key.value.vk
    return key.vk

def key_char(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.Key):
        return key.name #some keys don't have chars (shift etc.)
    return key.char


class KeyListener:
    def __init__(self, dataGenerator):
        self.dataGenerator = dataGenerator

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key is None:
            return
        self.dataGenerator.registerPress(
            key_code=key_code(key), timestamp=time.time(), key=key_char(key)
        )

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key is None:
            return
        self.dataGenerator.registerRelease(
            key_code=key_code(key), timestamp=time.time(), key=key_char(key)
        )

    def listen(self) -> None:
        print("Listening to keys...")
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()
