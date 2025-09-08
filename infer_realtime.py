# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch

from config import CKPT_DIR, FEATURE_DIM, SEQ_LEN, STATIC_THR, DYNAMIC_THR
from features import vectorize_landmarks
from models.lstm_model import LSTMClassifier
from utils import Debounce, Speaker, argmax_threshold

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def load_static_model():
    """Carrega RandomForest (estáticos)."""
    try:
        bundle = joblib.load(CKPT_DIR / "rf_static.joblib")
        return bundle["model"], bundle["labels"]
    except Exception:
        print("[Aviso] Modelo estático não encontrado. Seguindo sem RF.")
        return None, []


def load_dynamic_model(device):
    """Carrega LSTM (dinâmicos)."""
    try:
        ck = torch.load(CKPT_DIR / "lstm_dynamic.pt", map_location=device)
        model = LSTMClassifier(
            ck["input_dim"], ck["hidden_dim"], ck["num_layers"], num_classes=len(ck["labels"])
        )
        model.load_state_dict(ck["model_state"])
        model.eval()
        model.to(device)
        return model, ck["labels"]
    except Exception:
        print("[Aviso] Modelo dinâmico não encontrado. Seguindo sem LSTM.")
        return None, []


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def draw_landmarks(frame, results):
    """Desenha pose e mãos na imagem."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style()
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_styles.get_default_hand_connections_style()
        )


def put_text(frame, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def draw_sidebar(frame, prob_items, title="", highlight=None, width=300):
    """Barra lateral com probabilidades (lista de (label, prob))."""
    h, w = frame.shape[:2]
    x0, y0 = 0, 0
    x1 = min(width, w // 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, h), (30, 30, 60), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    pad = 12
    put_text(frame, title, (x0 + pad, 28), scale=0.9, color=(255, 255, 255), thickness=2)

    bar_left = x0 + pad
    bar_right = x1 - pad
    bar_w = bar_right - bar_left
    y = 60
    step = 26
    max_items = int((h - y - 20) / step)
    for label, prob in prob_items[:max_items]:
        col = (255, 230, 120) if label == highlight else (200, 220, 255)
        put_text(frame, f"{label}: {prob:.2f}", (bar_left, y), scale=0.6, color=col, thickness=2)
        by = y + 6
        cv2.rectangle(frame, (bar_left, by), (bar_right, by + 6), (80, 90, 120), -1)
        cv2.rectangle(frame, (bar_left, by), (bar_left + int(bar_w * float(np.clip(prob, 0, 1))), by + 6), (120, 200, 255), -1)
        y += step


def draw_progress(frame, progress, color=(180, 220, 255)):
    """Barra de progresso no rodapé (0..1) para a janela do dinâmico."""
    h, w = frame.shape[:2]
    pad = 10
    y0 = h - 20
    cv2.rectangle(frame, (pad, y0), (w - pad, y0 + 10), (70, 70, 70), -1)
    x_end = pad + int((w - 2 * pad) * float(np.clip(progress, 0, 1)))
    cv2.rectangle(frame, (pad, y0), (x_end, y0 + 10), color, -1)


def main():
    ap = argparse.ArgumentParser("LIBRAS realtime — LSTM (dinâmicos) + RF (estáticos)")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--static-thr", type=float, default=STATIC_THR)
    ap.add_argument("--dynamic-thr", type=float, default=DYNAMIC_THR)
    ap.add_argument("--title", type=str, default="Gesture Detection")
    ap.add_argument("--sidebar", type=int, default=300, help="Largura da barra lateral (px)")
    ap.add_argument("--no-tts", dest="no_tts", action="store_true", help="Desativa a fala (TTS)")
    ap.add_argument("--debounce-ms", type=int, default=350, help="Tempo para confirmar rótulo (ms)")
    ap.add_argument(
        "--phrase-map",
        type=str,
        default=str((Path(__file__).parent / "phrase_map.json")),
        help="Caminho para phrase_map.json",
    )
    args = ap.parse_args()

    # === Carrega mapeamento de rótulo -> frase (ex.: DIA -> "bom dia") ===
    PHRASE_MAP = {}
    pmpath = Path(args.phrase_map)
    if pmpath.exists():
        try:
            PHRASE_MAP = json.loads(pmpath.read_text(encoding="utf-8"))
        except Exception as e:
            print("[Aviso] phrase_map.json inválido:", e)
    # Fallbacks garantidos
    PHRASE_MAP.setdefault("DIA", "bom dia")
    PHRASE_MAP.setdefault("BOM_DIA", "bom dia")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_model, static_labels = load_static_model()
    dyn_model, dyn_labels = load_dynamic_model(device)
    if static_model is None and dyn_model is None:
        raise RuntimeError("Nenhum modelo encontrado. Treine RF (estático) e/ou LSTM (dinâmico) antes de rodar.")
    print("Modelos carregados.")

    tts_on = not args.no_tts
    speaker = Speaker()
    debounce = Debounce(hold_ms=args.debounce_ms)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam não encontrada.")

    # (Opcional) resoluções mais leves:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    buffer = []      # janela para dinâmicos
    transcript = []  # últimas palavras/frases confirmadas

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # === MediaPipe ===
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Extrai features (pose + mãos)
            pose = (
                np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark], dtype=np.float32)
                if results.pose_landmarks else None
            )
            left = (
                np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
                if results.left_hand_landmarks else None
            )
            right = (
                np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
                if results.right_hand_landmarks else None
            )
            feat = vectorize_landmarks(pose, left, right)

            # === Estático (RF) ===
            sp, static_idx, static_label = None, None, None
            if static_model is not None:
                sp = static_model.predict_proba([feat])[0] if hasattr(static_model, "predict_proba") else None
                if sp is not None:
                    static_idx = argmax_threshold(sp, args.static_thr)
                    static_label = static_labels[static_idx] if static_idx is not None else None

            # === Dinâmico (LSTM) ===
            buffer.append(feat)
            if len(buffer) > args.seq_len:
                buffer = buffer[-args.seq_len:]
            dynamic_label, dyn_probs = None, None
            if dyn_model is not None and len(buffer) == args.seq_len:
                xb = torch.from_numpy(np.array(buffer, dtype=np.float32)[None, ...]).to(device)
                with torch.no_grad():
                    logits = dyn_model(xb).cpu().numpy()
                    dyn_probs = softmax(logits)[0]

                    # 1) decisão padrão: argmax + threshold
                    d_idx = argmax_threshold(dyn_probs, args.dynamic_thr)
                    cand = dyn_labels[d_idx] if d_idx is not None else None

                    # 2) regra extra: só confirma gesto se vencer NONE por margem (ratio)
                    if cand is not None and 'NONE' in dyn_labels and cand != 'NONE':
                        p_cand = float(dyn_probs[d_idx])
                        p_none = float(dyn_probs[dyn_labels.index('NONE')])
                        ratio = p_cand / (p_cand + p_none + 1e-8)
                        if ratio < 0.55:  # >=55% do par (cand vs NONE)
                            cand = None

                    dynamic_label = cand

            # Decisão: dinâmico tem prioridade
            final_label = dynamic_label if dynamic_label is not None else static_label
            decided = debounce.update(final_label)

            # === Construção do display ===
            disp = frame.copy()
            draw_landmarks(disp, results)

            # Transcript (linha superior): só adiciona quando decisão válida (≠ NONE)
            if decided and decided != "NONE":
                token_raw = decided.replace("_", " ")
                token = PHRASE_MAP.get(decided, PHRASE_MAP.get(token_raw.upper(), token_raw))
                if not transcript or transcript[-1] != token:
                    transcript.append(token)
                    if tts_on:
                        try:
                            speaker.speak(token.lower())
                        except Exception:
                            pass
            phrase = " ".join(transcript[-8:])
            put_text(disp, phrase if phrase else " ", (20, 35), scale=1.0, color=(255, 255, 255), thickness=2)

            # Sidebar: probabilidades cruas (para depurar)
            items = []
            if sp is not None:
                items += list(zip(static_labels, sp.tolist()))
            if dyn_probs is not None:
                items += list(zip(dyn_labels, dyn_probs.tolist()))
            items.sort(key=lambda x: x[1], reverse=True)
            draw_sidebar(disp, items, title=args.title, highlight=final_label, width=args.sidebar)

            # Barra de progresso da sequência (dinâmico)
            prog = len(buffer) / max(1, args.seq_len)
            draw_progress(disp, prog)

            # Título grande: MOSTRA SOMENTE quando decisão válida (≠ NONE)
            if decided and decided != "NONE":
                big = PHRASE_MAP.get(decided, PHRASE_MAP.get(decided.replace("_", " ").upper(), decided))
                put_text(disp, big, (args.sidebar + 16, 70), scale=1.2, color=(120, 255, 120), thickness=3)

            cv2.imshow("Gesture Detection", disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                transcript = []
            elif key in (ord("s"), ord("S")):
                tts_on = not tts_on

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
