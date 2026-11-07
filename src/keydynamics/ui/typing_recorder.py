"""Record textbox key events and build live keystroke samples."""

import time
from queue import Empty, Queue
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QEvent, QObject, Signal

from keydynamics.listener.datagenerator import DataGenerator
from keydynamics.models.model import KeyModel, SampleProps


class TypingRecorder(QObject):
    model_changed = Signal(object, int)

    def __init__(self, textbox):
        super().__init__(textbox)
        self.textbox = textbox
        self.events = Queue()
        self.generator = DataGenerator(
            participant_id=0, session_id=uuid4(), memory=self.events, sample_timeout=3
        )
        self.live_model = KeyModel(sample_props=[])
        self.sample_events = {}
        self.live_props = {}
        self.completed = 0
        textbox.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.textbox and event.type() == QEvent.Type.FocusOut:
            # Releases outside this widget may never arrive. Start a fresh sample.
            self.generator.pressed_keys.clear()
            self.generator.sample_id += 1
            self.update_sample()
        if (
            watched is self.textbox
            and event.type()
            in (
                QEvent.Type.KeyPress,
                QEvent.Type.KeyRelease,
            )
            and not event.isAutoRepeat()
        ):
            key_code = event.nativeScanCode() or event.key()
            timestamp = time.perf_counter()
            text = event.text() or None

            if event.type() == QEvent.Type.KeyPress:
                self.generator.registerPress(key_code, timestamp, text)
            else:
                self.generator.registerRelease(key_code, timestamp, text)
                self.update_sample()

        # Let the text box process the event normally.
        return super().eventFilter(watched, event)

    def drain_queue(self):
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            self.sample_events.setdefault(event.sample_id, []).append(event)

    def update_sample(self):
        self.drain_queue()

        sample_id = self.generator.sample_id
        sample = self.sample_events.get(sample_id, [])
        completed = 0
        if sample:
            presses = {event.press_id for event in sample if event.event == "PRESS"}
            releases = {event.press_id for event in sample if event.event == "RELEASE"}
            completed = len(presses & releases)
            props = SampleProps.from_sample(sample)
            if props is not None and np.isfinite(props.to_np()).all():
                self.live_props[sample_id] = props

        self.live_model = KeyModel(
            sample_props=[self.live_props[i] for i in sorted(self.live_props)]
        )
        self.completed = completed
        self.model_changed.emit(self.live_model, completed)
