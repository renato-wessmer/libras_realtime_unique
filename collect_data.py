# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from config import STATIC_DIR, DYNAMIC_DIR, SEQ_LEN, FEATURE_DIM
from features import vectorize_landmarks

mp_holistic = mp.solutions.holistic

def extract_from_frame(holistic, frame_bgr):
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    # Coleta pose (33), mãos esquerda e direita (21 cada)
    pose = None
    if results.pose_landmarks:
        pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark], dtype=np.float32)
    left = None
    if results.left_hand_landmarks:
        left = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
    right = None
    if results.right_hand_landmarks:
        right = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
    feat = vectorize_landmarks(pose, left, right)
    return feat

def collect_static(label: str, num: int):
    out_dir = STATIC_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam não encontrada.")
    print("Coleta ESTÁTICA: Pressione SPACE para salvar amostra, Q para sair.")
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as holistic:
        saved = 0
        while saved < num:
            ok, frame = cap.read()
            if not ok:
                break
            feat = extract_from_frame(holistic, frame)
            disp = frame.copy()
            cv2.putText(disp, f"ESTATICO {label} {saved}/{num}  [SPACE=Salvar  Q=Sair]", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Coleta Estatica", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                idx = len(list(out_dir.glob("*.npy"))) + 1
                np.save(str(out_dir / f"{idx:05d}.npy"), feat)
                saved += 1
            elif key in (ord('q'), ord('Q')):
                break
    cap.release()
    cv2.destroyAllWindows()

def collect_dynamic(label: str, num: int, seq_len: int):
    out_dir = DYNAMIC_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam não encontrada.")
    print("Coleta DINÂMICA: SPACE inicia/para gravação de uma sequência; Q para sair.")
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as holistic:
        sequences = 0
        recording = False
        buffer = []
        seq_id = len(list(out_dir.glob("*"))) + 1
        seq_dir = out_dir / f"{seq_id:04d}"
        while sequences < num:
            ok, frame = cap.read()
            if not ok:
                break
            feat = extract_from_frame(holistic, frame)
            disp = frame.copy()
            status = "REC" if recording else "IDLE"
            cv2.putText(disp, f"DINAMICO {label} {sequences}/{num} ({status})  [SPACE=REC  Q=Sair]", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if recording else (0,255,255), 2)
            cv2.imshow("Coleta Dinamica", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                recording = not recording
                if recording:
                    buffer = []
                    seq_dir = out_dir / f"{seq_id:04d}"
                    seq_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # salvar sequência (pode truncar/preencher para seq_len)
                    arr = np.stack(buffer, axis=0) if buffer else np.zeros((0, FEATURE_DIM), dtype=np.float32)
                    if len(buffer) >= 2:
                        # ajusta para tamanho fixo
                        if arr.shape[0] > seq_len:
                            arr = arr[:seq_len]
                        elif arr.shape[0] < seq_len:
                            pad = np.tile(arr[-1], (seq_len - arr.shape[0], 1))
                            arr = np.vstack([arr, pad])
                        # salva como frames npy individuais para facilitar debug
                        for i in range(arr.shape[0]):
                            np.save(str(seq_dir / f"{i:03d}.npy"), arr[i])
                        sequences += 1
                        seq_id += 1
            elif key in (ord('q'), ord('Q')):
                break

            if recording:
                buffer.append(feat)
                if len(buffer) >= seq_len:
                    # auto-stop para sequência de tamanho fixo
                    recording = False
                    arr = np.stack(buffer, axis=0)
                    for i in range(arr.shape[0]):
                        np.save(str(seq_dir / f"{i:03d}.npy"), arr[i])
                    sequences += 1
                    seq_id += 1
                    buffer = []

    cap.release()
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "dynamic"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--num", type=int, default=100, help="Qtde de amostras/sequências")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    if args.mode == "static":
        collect_static(args.label, args.num)
    else:
        collect_dynamic(args.label, args.num, args.seq_len)

if __name__ == "__main__":
    main()
