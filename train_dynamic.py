# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from config import DYNAMIC_DIR, CKPT_DIR, FEATURE_DIM, SEQ_LEN, load_labels
from models.lstm_model import LSTMClassifier

class SeqDataset(Dataset):
    def __init__(self, root: Path, labels: list, seq_len: int):
        self.samples = []
        self.labels = labels
        self.seq_len = seq_len
        for yi, label in enumerate(labels):
            label_dir = root / label
            for seq_dir in sorted([p for p in label_dir.glob("*") if p.is_dir()]):
                frames = sorted(list(seq_dir.glob("*.npy")))
                if len(frames) == 0:
                    continue
                seq = [np.load(f) for f in frames]
                arr = np.stack(seq, axis=0)
                # Ajusta tamanho
                if arr.shape[0] > seq_len:
                    arr = arr[:seq_len]
                elif arr.shape[0] < seq_len:
                    pad = np.tile(arr[-1], (seq_len - arr.shape[0], 1))
                    arr = np.vstack([arr, pad])
                self.samples.append((arr.astype(np.float32), yi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def split_dataset(ds, val_ratio=0.2, seed=42):
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    from torch.utils.data import Subset
    return Subset(ds, tr_idx), Subset(ds, val_idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    _, dynamic_labels = load_labels()
    ds = SeqDataset(DYNAMIC_DIR, dynamic_labels, args.seq_len)
    if len(ds) == 0:
        raise RuntimeError("Nenhuma sequência dinâmica encontrada.")
    tr, va = split_dataset(ds, val_ratio=0.2, seed=42)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=FEATURE_DIM, hidden_dim=args.hidden, num_layers=args.layers, num_classes=len(dynamic_labels))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validação
        model.eval()
        ys, yhat = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                yhat.extend(pred)
                ys.extend(yb.numpy().tolist())
        acc = accuracy_score(ys, yhat)
        print(f"Epoch {epoch:02d} | Val Acc: {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "labels": dynamic_labels,
                "input_dim": FEATURE_DIM,
                "hidden_dim": args.hidden,
                "num_layers": args.layers
            }, CKPT_DIR / "lstm_dynamic.pt")
            print("Modelo salvo em models_ckpt/lstm_dynamic.pt")

    print("Treino concluído. Melhor acc:", best_acc)

if __name__ == "__main__":
    main()
