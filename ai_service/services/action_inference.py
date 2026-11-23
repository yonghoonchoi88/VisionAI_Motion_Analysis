"""
사람별 동작 추론 (action_inference.py)

역할
    pose_pipeline.py가 만든 '사람(track_id)별 Window 묶음'을 Random Forest에 넣고,
    사람마다 최종 Action · Confidence 하나로 정리한다.

입력 형태
    windows_by_track = {track_id: ndarray}
        ndarray shape = (n_windows, 1140)            ← 1140 = 30 steps × 38, feature_columns.json 순서
                     또는 (n_windows, 30, 38)         ← 자동으로 (n_windows, 1140)으로 펼침

집계 방식
    한 사람에게 Window가 여러 개 있으면, Window별 확률(predict_proba)을 '평균'낸 뒤
    가장 높은 Class를 최종 Action으로 고른다. (Window 하나의 순간 오판에 덜 흔들림)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from services.model_loader import ModelBundle

# Confidence가 이 값보다 낮으면 화면에 '판단 불확실'로 표시한다.
# 7일차 day07_realtime.json의 confidence_threshold(0.55)와 같은 값. (실제 값은 project.json에서 전달)
DEFAULT_UNCERTAIN_THRESHOLD = 0.55


@dataclass(frozen=True)
class PersonPrediction:
    track_id: int
    action_label: str                  # 예: 'wave'
    action_id: str                     # 예: 'A006'
    confidence: float                  # 0~1, 평균 확률 중 최댓값
    is_uncertain: bool                 # confidence < threshold
    window_count: int                  # 판단에 사용한 Window 수
    probabilities: dict[str, float]    # Class별 평균 확률 (디버깅·대시보드용)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_matrix(bundle: ModelBundle, windows: np.ndarray, track_id: int) -> np.ndarray:
    """입력을 (n_windows, 1140) float 행렬로 정규화하고 형태·값을 검사한다."""
    x = np.asarray(windows, dtype=np.float64)

    if x.ndim == 3:  # (n, 30, 38) → (n, 1140). C-order로 펼치면 t00의 38개 → t01의 38개 … 순서가 된다.
        if x.shape[1:] != (bundle.sequence_steps, bundle.features_per_step):
            raise ValueError(
                f"track {track_id}: 3D 입력은 (n, {bundle.sequence_steps}, {bundle.features_per_step})여야 합니다. 받은 shape={x.shape}"
            )
        x = x.reshape(x.shape[0], -1)

    if x.ndim != 2 or x.shape[1] != bundle.n_features:
        raise ValueError(f"track {track_id}: 입력은 (n, {bundle.n_features})여야 합니다. 받은 shape={x.shape}")
    if x.shape[0] == 0:
        raise ValueError(f"track {track_id}: Window가 0개입니다.")
    if not np.isfinite(x).all():
        raise ValueError(f"track {track_id}: NaN/Inf가 포함되어 있습니다. (결측 보간은 pose_pipeline에서 처리)")
    return x


def _predict_proba(bundle: ModelBundle, x: np.ndarray) -> np.ndarray:
    """학습 시 DataFrame을 썼다면 같은 컬럼 이름으로 감싸서 넣는다 (순서·경고 방지)."""
    if bundle.expects_dataframe:
        import pandas as pd  # 필요할 때만 import
        x = pd.DataFrame(x, columns=list(bundle.feature_columns))
    return bundle.model.predict_proba(x)


def predict_track(
    bundle: ModelBundle,
    track_id: int,
    windows: np.ndarray,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
) -> PersonPrediction:
    """한 사람의 Window 묶음 → 최종 예측 1건."""
    x = _to_matrix(bundle, windows, track_id)
    proba = _predict_proba(bundle, x)          # shape (n_windows, 6), 열 순서 = bundle.classes
    mean_proba = proba.mean(axis=0)            # Window 평균 → (6,)

    best = int(np.argmax(mean_proba))
    label = bundle.classes[best]               # ★ index → 이름은 반드시 bundle.classes(= model.classes_)로 변환
    confidence = float(mean_proba[best])

    return PersonPrediction(
        track_id=int(track_id),
        action_label=label,
        action_id=bundle.action_id(label),
        confidence=round(confidence, 4),
        is_uncertain=confidence < uncertain_threshold,
        window_count=int(x.shape[0]),
        probabilities={c: round(float(p), 4) for c, p in zip(bundle.classes, mean_proba)},
    )


def predict_tracks(
    bundle: ModelBundle,
    windows_by_track: Mapping[int, np.ndarray],
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
) -> list[PersonPrediction]:
    """여러 사람 → 사람별 예측 리스트 (track_id 오름차순). 사람이 없으면 빈 리스트."""
    return [
        predict_track(bundle, tid, windows_by_track[tid], uncertain_threshold)
        for tid in sorted(windows_by_track)
    ]
