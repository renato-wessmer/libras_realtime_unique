# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = DATA_DIR / "static"
DYNAMIC_DIR = DATA_DIR / "dynamic"
CKPT_DIR = BASE_DIR / "models_ckpt"

LABEL_MAP_FILE = BASE_DIR / "label_map.json"

# Parâmetros padrão
SEQ_LEN = 30
FEATURE_DIM = 225  # 21*3*2 (mãos) + 33*3 (pose) = 225
STATIC_THR = 0.6
DYNAMIC_THR = 0.6

def load_labels():
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        lm = json.load(f)
    static_labels = lm.get("static", [])
    dynamic_labels = lm.get("dynamic", [])
    return static_labels, dynamic_labels
