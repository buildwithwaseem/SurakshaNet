"""
SurakshaNet - Audio Alert Module
Plays a spoken/beep alert when a PPE breach is detected, with a cooldown
so it doesn't fire on every frame and lock up the pipeline.
"""

import threading
import time

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class SirenAlert:
    def __init__(self, cooldown_seconds: float = 5.0, message: str = "Warning. Safety violation detected."):
        """
        cooldown_seconds: minimum gap between two consecutive alerts.
        message: spoken text when a breach occurs.
        """
        self.cooldown = cooldown_seconds
        self.message = message
        self._last_triggered = 0.0
        self._lock = threading.Lock()

        if _TTS_AVAILABLE:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 165)
        else:
            self._engine = None
            print("WARNING: pyttsx3 not installed — siren will fall back to console beep.")

    def _speak(self):
        if self._engine:
            try:
                self._engine.say(self.message)
                self._engine.runAndWait()
            except Exception as e:
                print(f"ERROR: TTS error: {e}")
        else:
            # Fallback: terminal bell
            print("\aSAFETY ALERT")

    def trigger(self, force: bool = False):
        """
        Call this when a breach is detected. Internally throttled by cooldown.
        Runs in a background thread so it never blocks the video loop.
        """
        now = time.time()
        with self._lock:
            if not force and (now - self._last_triggered) < self.cooldown:
                return  # still in cooldown, skip
            self._last_triggered = now

        threading.Thread(target=self._speak, daemon=True).start()
