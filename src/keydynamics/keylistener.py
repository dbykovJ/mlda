from pynput import keyboard


class Keylistener:
    def on_press(self, key):
        print(f"registering key {key}")

    def on_release(self, key):
        print(f"releasing {key}")

    def listen(self):
        print("listeing to keys...")
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()
