# WATERMARK: bda58d49-04d8-49ab-9657-eec961b97cc9 :: 2025-09-07T18:50:21.115875
from typing import Tuple
import numpy as np

def _normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normaliza pontos (x, y, z) removendo translação e escala, para robustez.
    points: (N, 3), com zeros quando ausente.
    - Centraliza no centroide
    - Escala pela distância média aos ombros (se disponível) ou norma média
    """
    pts = points.copy()
    if pts.size == 0:
        return pts
    # Máscara de pontos válidos (não zero)
    mask = ~(np.all(pts == 0.0, axis=1))
    if not np.any(mask):
        return pts
    valid = pts[mask]
    center = valid.mean(axis=0, keepdims=True)
    pts[mask] = valid - center

    # Escala pela norma média
    norms = np.linalg.norm(pts[mask], axis=1)
    scale = norms.mean() if norms.mean() > 1e-6 else 1.0
    pts[mask] /= scale
    return pts

def vectorize_landmarks(pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    Concatena pose (33x3) + mãos (21x3 cada) -> (225,)
    Substitui ausentes por zeros e normaliza.
    """
    if pose is None:
        pose = np.zeros((33, 3), dtype=np.float32)
    if left_hand is None:
        left_hand = np.zeros((21, 3), dtype=np.float32)
    if right_hand is None:
        right_hand = np.zeros((21, 3), dtype=np.float32)

    pose_n = _normalize_points(pose)
    lh_n = _normalize_points(left_hand)
    rh_n = _normalize_points(right_hand)

    feat = np.concatenate([pose_n.flatten(), lh_n.flatten(), rh_n.flatten()], axis=0).astype(np.float32)
    return feat  # (225,)

def ewma_probs(probs: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Suaviza uma sequência de vetores de probabilidade (T, C) com EWMA."""
    if probs.ndim != 2:
        return probs
    out = np.zeros_like(probs)
    out[0] = probs[0]
    for t in range(1, probs.shape[0]):
        out[t] = alpha * probs[t] + (1 - alpha) * out[t - 1]
    return out
