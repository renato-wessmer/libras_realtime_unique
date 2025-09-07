# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
from typing import Optional, Tuple, Dict
import numpy as np
import json
import time
import pyttsx3

def now_ms() -> int:
    return int(time.time() * 1000)

class Debounce:
    def __init__(self, hold_ms: int = 600):
        self.hold_ms = hold_ms
        self.last_label = None
        self.start_ms = 0

    def update(self, label: Optional[str]) -> Optional[str]:
        t = now_ms()
        if label != self.last_label:
            self.last_label = label
            self.start_ms = t
            return None
        if label is None:
            return None
        if t - self.start_ms >= self.hold_ms:
            return label
        return None

class Speaker:
    def __init__(self, rate: int = 180, voice_contains: str = "braz"):
        self.engine = pyttsx3.init()
        # Tenta achar voz pt-BR
        try:
            for v in self.engine.getProperty("voices"):
                name = (v.name or "").lower()
                lang = ",".join(getattr(v, "languages", [])).lower()
                if voice_contains in name or "pt" in lang:
                    self.engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass
        self.engine.setProperty("rate", rate)

    def speak(self, text: str):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass

def load_label_map(path: str) -> Dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def argmax_threshold(probs: np.ndarray, thr: float) -> Optional[int]:
    idx = int(np.argmax(probs))
    if probs[idx] >= thr:
        return idx
    return None
