# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875

import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import random
from features import vectorize_landmarks
from config import STATIC_DIR, DYNAMIC_DIR, FEATURE_DIM, SEQ_LEN

mp_holistic = mp.solutions.holistic

def save_static_sample(label_dir: Path, feat: np.ndarray):
    label_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(label_dir.glob('*.npy'))) + 1
    np.save(str(label_dir / f"{idx:05d}.npy"), feat)

def save_dynamic_seq(label_dir: Path, seq: np.ndarray):
    label_dir.mkdir(parents=True, exist_ok=True)
    sid = len([p for p in label_dir.glob('*') if p.is_dir()]) + 1
    seq_dir = label_dir / f"{sid:04d}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i in range(seq.shape[0]):
        np.save(str(seq_dir / f"{i:03d}.npy"), seq[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Caminho do vídeo de entrada')
    ap.add_argument('--mode', choices=['static', 'dynamic'], required=True)
    ap.add_argument('--label', required=True, help='Nome da classe POSITIVA (ex.: MEU)')
    ap.add_argument('--neg-label', default='NONE', help='Classe negativa/ruído')
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN)
    ap.add_argument('--write-neg', action='store_true', help='Gerar negativos a partir de trechos não marcados')
    ap.add_argument('--neg-rate', type=float, default=0.15, help='Probabilidade de salvar um frame/seq negativo')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Não consegui abrir o vídeo.')

    pos_on = False
    buf = []  # para dinâmico
    total_pos_static = 0
    total_neg_static = 0
    total_pos_seq = 0
    total_neg_seq = 0

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                # flush final de buffer dinâmico (se pos_on e suficiente)
                if args.mode == 'dynamic' and pos_on and len(buf) >= args.seq_len:
                    arr = np.stack(buf[:args.seq_len], axis=0)
                    save_dynamic_seq(DYNAMIC_DIR / args.label, arr.astype(np.float32))
                    total_pos_seq += 1
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

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

            disp = frame.copy()
            cv2.putText(disp, f"{Path(args.video).name}  [S=toggle positivo | Q=sair]", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            status = 'POS' if pos_on else 'NEG'
            cv2.putText(disp, f"{status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if pos_on else (0,200,255), 2)
            if args.mode == 'dynamic':
                prog = len(buf) / max(1, args.seq_len)
                h, w = disp.shape[:2]
                y0 = h - 18
                pad = 10
                cv2.rectangle(disp, (pad, y0), (w - pad, y0 + 10), (60,60,60), -1)
                cv2.rectangle(disp, (pad, y0), (pad + int((w-2*pad)*prog), y0 + 10), (120,220,255), -1)
            cv2.imshow('Ingest from Video', disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                # alterna estado positivo/negativo
                if args.mode == 'dynamic' and pos_on and len(buf) >= args.seq_len:
                    # ao sair do segmento positivo, salva sequência
                    arr = np.stack(buf[:args.seq_len], axis=0)
                    save_dynamic_seq(DYNAMIC_DIR / args.label, arr.astype(np.float32))
                    total_pos_seq += 1
                    buf = []
                pos_on = not pos_on

            # gravação
            if args.mode == 'static':
                if pos_on:
                    save_static_sample(STATIC_DIR / args.label, feat)
                    total_pos_static += 1
                elif args.write_neg and random.random() < args.neg_rate:
                    save_static_sample(STATIC_DIR / args.neg_label, feat)
                    total_neg_static += 1

            else:  # dynamic
                buf.append(feat)
                if len(buf) > args.seq_len:
                    buf = buf[-args.seq_len:]
                if pos_on and len(buf) == args.seq_len:
                    arr = np.stack(buf, axis=0)
                    save_dynamic_seq(DYNAMIC_DIR / args.label, arr.astype(np.float32))
                    total_pos_seq += 1
                    buf = []  # reinicia para sequências não sobrepostas
                elif (not pos_on) and args.write_neg and len(buf) == args.seq_len and random.random() < args.neg_rate:
                    arr = np.stack(buf, axis=0)
                    save_dynamic_seq(DYNAMIC_DIR / args.neg_label, arr.astype(np.float32))
                    total_neg_seq += 1
                    buf = []

    cap.release()
    cv2.destroyAllWindows()

    print('Concluído.')
    if args.mode == 'static':
        print(f'Positivos (static): {total_pos_static} | Negativos (static): {total_neg_static}')
    else:
        print(f'Positivos (dynamic): {total_pos_seq} | Negativos (dynamic): {total_neg_seq}')

if __name__ == '__main__':
    main()
