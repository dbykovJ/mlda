from pynput import keyboard


def on_press(key):
    print(f"registering key {key}")


def on_release(key):
    print(f"releasing {key}")


def main() -> None:
    print("Hello from keydynamics!")
    print("listeing to keys...")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
