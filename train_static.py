# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from config import STATIC_DIR, CKPT_DIR, FEATURE_DIM, load_labels

def load_static_dataset():
    X, y, names = [], [], []
    static_labels, _ = load_labels()
    for yi, label in enumerate(static_labels):
        for npy_file in (STATIC_DIR / label).glob("*.npy"):
            feat = np.load(npy_file)
            if feat.shape[0] != FEATURE_DIM:
                continue
            X.append(feat)
            y.append(yi)
            names.append(label)
    if not X:
        raise RuntimeError("Nenhuma amostra estática encontrada.")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, static_labels

def main():
    X, y, labels = load_static_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print("Acurácia (holdout):", acc)
    print(classification_report(yte, ypred, target_names=labels, digits=3))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": labels}, CKPT_DIR / "rf_static.joblib")
    print("Modelo salvo em models_ckpt/rf_static.joblib")

if __name__ == "__main__":
    main()
