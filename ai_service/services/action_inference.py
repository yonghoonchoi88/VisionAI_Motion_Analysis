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
    - sliding / whole_track : Window별 확률(predict_proba)의 평균
    - hybrid                : 0번 행(전체 구간) 확률 × 0.5 + 나머지 sliding Window 평균 × 0.5
                              (전체 구간 = 6일차 학습과 같은 입력, Window 평균 = 순간 오판 완화)

판단 불확실 (DecisionRule)
    confidence < min_confidence  또는  margin(1등 - 2등 확률) < min_margin  → is_uncertain=True
    6 Class 모델에서 무작위 수준은 1/6 ≈ 0.167. 소량 데이터(S01 · 6 clip)로 학습한 RF는 새 사람에게
    0.2~0.35 정도의 확률을 내므로, 1등 확률 하나보다 "2등과 얼마나 벌어졌는지"가 더 많은 정보를 준다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from services.model_loader import ModelBundle

# 기본값 = 예전 동작(7일차 day07_realtime.json의 confidence_threshold 0.55, margin 미사용).
# 실제 값은 project.json의 decision 블록에서 전달된다.
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MIN_MARGIN = 0.0
HYBRID_WHOLE_WEIGHT = 0.5


@dataclass(frozen=True)
class DecisionRule:
    """'판단 불확실' 기준. 두 조건 중 하나라도 걸리면 불확실."""

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_margin: float = DEFAULT_MIN_MARGIN

    def is_uncertain(self, confidence: float, margin: float) -> bool:
        return confidence < self.min_confidence or margin < self.min_margin

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PersonPrediction:
    track_id: int
    action_label: str                  # 예: 'wave' (불확실이어도 1등 Label은 그대로 둔다 — 화면에서 '불확실' 표시)
    action_id: str                     # 예: 'A006'
    confidence: float                  # 0~1, 집계 확률 중 최댓값
    margin: float                      # 1등 확률 - 2등 확률 (0이면 동률)
    runner_up_label: str               # 2등 Label — '무엇과 헷갈렸는지' 설명용
    is_uncertain: bool                 # DecisionRule에 걸리면 True
    window_count: int                  # 판단에 사용한 Window 수 (hybrid는 전체 구간 1개 포함)
    probabilities: dict[str, float]    # Class별 집계 확률 (디버깅·화면 막대그래프용)

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


def _aggregate(proba: np.ndarray, window_mode: str) -> np.ndarray:
    """Window별 확률 (n, 6) → 사람 1명의 확률 (6,)."""
    if window_mode == "hybrid" and proba.shape[0] >= 2:
        return HYBRID_WHOLE_WEIGHT * proba[0] + (1 - HYBRID_WHOLE_WEIGHT) * proba[1:].mean(axis=0)
    return proba.mean(axis=0)


def _as_rule(rule: DecisionRule | float | None) -> DecisionRule:
    """예전 호출 방식(임계값 float 하나)도 받아 준다."""
    if rule is None:
        return DecisionRule()
    if isinstance(rule, DecisionRule):
        return rule
    return DecisionRule(min_confidence=float(rule))


def predict_track(
    bundle: ModelBundle,
    track_id: int,
    windows: np.ndarray,
    rule: DecisionRule | float | None = None,
    window_mode: str = "sliding",
) -> PersonPrediction:
    """한 사람의 Window 묶음 → 최종 예측 1건."""
    rule = _as_rule(rule)
    x = _to_matrix(bundle, windows, track_id)
    proba = _predict_proba(bundle, x)          # shape (n_windows, 6), 열 순서 = bundle.classes
    final = _aggregate(proba, window_mode)     # (6,)

    order = np.argsort(final)[::-1]            # 확률 큰 순서
    best, second = int(order[0]), int(order[1])
    label = bundle.classes[best]               # ★ index → 이름은 반드시 bundle.classes(= model.classes_)로 변환
    confidence = float(final[best])
    margin = float(final[best] - final[second])

    return PersonPrediction(
        track_id=int(track_id),
        action_label=label,
        action_id=bundle.action_id(label),
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        runner_up_label=bundle.classes[second],
        is_uncertain=rule.is_uncertain(confidence, margin),
        window_count=int(x.shape[0]),
        probabilities={c: round(float(p), 4) for c, p in zip(bundle.classes, final)},
    )


def predict_tracks(
    bundle: ModelBundle,
    windows_by_track: Mapping[int, np.ndarray],
    rule: DecisionRule | float | None = None,
    modes_by_track: Mapping[int, str] | None = None,
) -> list[PersonPrediction]:
    """여러 사람 → 사람별 예측 리스트 (track_id 오름차순). 사람이 없으면 빈 리스트."""
    modes = modes_by_track or {}
    return [
        predict_track(bundle, tid, windows_by_track[tid], rule, modes.get(tid, "sliding"))
        for tid in sorted(windows_by_track)
    ]
