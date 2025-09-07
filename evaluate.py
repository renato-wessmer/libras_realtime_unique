# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
from pathlib import Path
import numpy as np
import joblib
import torch
from config import STATIC_DIR, DYNAMIC_DIR, CKPT_DIR, FEATURE_DIM, SEQ_LEN, load_labels
from sklearn.metrics import accuracy_score, classification_report
from models.lstm_model import LSTMClassifier

def eval_static():
    bundle = joblib.load(CKPT_DIR / "rf_static.joblib")
    clf = bundle["model"]
    labels = bundle["labels"]
    X, y = [], []
    for yi, label in enumerate(labels):
        for f in (STATIC_DIR / label).glob("*.npy"):
            X.append(np.load(f))
            y.append(yi)
    if not X:
        print("Sem dados estáticos p/ avaliar.")
        return
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    yhat = clf.predict(X)
    print("== Estáticos ==")
    print("Acc:", accuracy_score(y, yhat))
    print(classification_report(y, yhat, target_names=labels, digits=3))

def eval_dynamic():
    ck = torch.load(CKPT_DIR / "lstm_dynamic.pt", map_location="cpu")
    labels = ck["labels"]
    model = LSTMClassifier(ck["input_dim"], ck["hidden_dim"], ck["num_layers"], num_classes=len(labels))
    model.load_state_dict(ck["model_state"])
    model.eval()

    X, y = [], []
    for yi, label in enumerate(labels):
        label_dir = DYNAMIC_DIR / label
        for seq_dir in sorted([p for p in label_dir.glob("*") if p.is_dir()]):
            frames = sorted(list(seq_dir.glob("*.npy")))
            if len(frames) == 0:
                continue
            seq = [np.load(f) for f in frames]
            arr = np.stack(seq, axis=0).astype(np.float32)
            if arr.shape[0] > SEQ_LEN:
                arr = arr[:SEQ_LEN]
            elif arr.shape[0] < SEQ_LEN:
                pad = np.tile(arr[-1], (SEQ_LEN - arr.shape[0], 1))
                arr = np.vstack([arr, pad])
            X.append(arr[None, ...])
            y.append(yi)
    if not X:
        print("Sem dados dinâmicos p/ avaliar.")
        return
    X = np.concatenate(X, axis=0)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        yhat = logits.argmax(dim=1).numpy()
    print("== Dinâmicos ==")
    print("Acc:", accuracy_score(y, yhat))
    print(classification_report(y, yhat, target_names=labels, digits=3))

if __name__ == "__main__":
    eval_static()
    eval_dynamic()
