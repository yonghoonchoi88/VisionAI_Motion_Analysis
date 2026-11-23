"""판단 불확실 기준(Confidence + margin) · hybrid 집계 · 전략 동시 비교 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from services.action_inference import DecisionRule, predict_track
from services.pose_pipeline import PipelineConfig


class FixedProbaModel:
    """predict_proba가 Window 순서대로 정해진 확률을 돌려주는 가짜 분류기."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float)

    def predict_proba(self, x):
        return self.rows[: len(x)]


@pytest.fixture
def fixed_bundle(synthetic_bundle):
    """진짜 규격(1140 · 6 Class)은 그대로, 모델만 확률을 고정한 가짜로 바꾼다."""
    def _make(rows):
        return SimpleNamespace(
            model=FixedProbaModel(rows),
            classes=synthetic_bundle.classes,
            n_features=synthetic_bundle.n_features,
            sequence_steps=synthetic_bundle.sequence_steps,
            features_per_step=synthetic_bundle.features_per_step,
            feature_columns=synthetic_bundle.feature_columns,
            expects_dataframe=False,
            action_id=synthetic_bundle.action_id,
        )
    return _make


# classes(알파벳순) = bend_return, drink_return, leg_raise_lower, sit_stand, walk_turn_walk, wave
WAVE_CLEAR = [0.12, 0.14, 0.19, 0.12, 0.08, 0.35]      # wave 1등, 2등과 0.16 차이
WAVE_CLOSE = [0.10, 0.20, 0.18, 0.12, 0.18, 0.22]      # wave 1등, drink_return과 0.02 차이


def test_rule_both_conditions():
    rule = DecisionRule(min_confidence=0.25, min_margin=0.05)
    assert rule.is_uncertain(0.20, 0.10)        # 확률이 낮음
    assert rule.is_uncertain(0.30, 0.02)        # 1·2등이 붙어 있음
    assert not rule.is_uncertain(0.30, 0.10)


def test_margin_and_runner_up(fixed_bundle):
    p = predict_track(fixed_bundle([WAVE_CLOSE]), 1, np.zeros((1, 1140)), DecisionRule(0.2, 0.05))
    assert p.action_label == "wave" and p.runner_up_label == "drink_return"
    assert p.margin == pytest.approx(0.02)
    assert p.is_uncertain                        # 확률(0.22)은 넘지만 margin이 작아 불확실


def test_clear_prediction_is_confirmed_with_margin_rule(fixed_bundle):
    """0.55 기준이면 불확실이던 0.35도, 2등과 충분히 벌어지면 확정할 수 있다."""
    windows = np.zeros((1, 1140))
    assert predict_track(fixed_bundle([WAVE_CLEAR]), 1, windows, DecisionRule(0.55, 0.0)).is_uncertain
    assert not predict_track(fixed_bundle([WAVE_CLEAR]), 1, windows, DecisionRule(0.25, 0.10)).is_uncertain


def test_legacy_float_threshold_still_works(fixed_bundle):
    p = predict_track(fixed_bundle([WAVE_CLEAR]), 1, np.zeros((1, 1140)), 0.30)
    assert not p.is_uncertain and p.margin > 0


def test_hybrid_weights_whole_track_half(fixed_bundle):
    """hybrid: 0번 행(전체 구간) 0.5 + 나머지 Window 평균 0.5 — 단순 평균과 결과가 달라야 한다."""
    whole = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    w1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bundle = fixed_bundle([whole, w1, w2])
    hybrid = predict_track(bundle, 1, np.zeros((3, 1140)), window_mode="hybrid")
    plain = predict_track(bundle, 1, np.zeros((3, 1140)), window_mode="sliding")
    assert hybrid.probabilities["wave"] == pytest.approx(0.5)
    assert plain.probabilities["wave"] == pytest.approx(1 / 3, abs=1e-4)
    assert plain.action_label == "bend_return"


def test_run_strategies_reuses_one_pose_pass(make_pipeline, make_video):
    """YOLO는 영상당 1번만 호출하고, 4개 전략 결과를 모두 만든다 (A/B 비교를 같은 입력으로)."""
    pipe = make_pipeline(lambda i: {1: 300})
    video = make_video(seconds=8)
    outs = pipe.run_strategies(video, ["sliding", "whole_track", "auto", "hybrid"])
    assert pipe.model.calls == 8 * 30
    assert outs["whole_track"].tracks[0].windows.shape == (1, 1140)
    assert outs["auto"].tracks[0].window_mode == "whole_track"            # 8초 ≤ 12초
    n_sliding = outs["sliding"].tracks[0].windows.shape[0]
    assert outs["hybrid"].tracks[0].window_mode == "hybrid"
    assert outs["hybrid"].tracks[0].windows.shape[0] == n_sliding + 1      # 전체 구간 1 + sliding


def test_run_strategies_rejects_unknown(make_pipeline, make_video):
    with pytest.raises(ValueError):
        make_pipeline(lambda i: {1: 300}).run_strategies(make_video(seconds=1), ["magic"])


def test_decision_env_override(monkeypatch, tmp_path):
    from tests.conftest import AI_ROOT

    monkeypatch.setenv("MIN_CONFIDENCE", "0.27")
    monkeypatch.setenv("MIN_MARGIN", "0.06")
    cfg = PipelineConfig.load(AI_ROOT / "project.json")
    assert (cfg.min_confidence, cfg.min_margin) == (0.27, 0.06)


def test_decision_out_of_range_rejected(monkeypatch):
    from tests.conftest import AI_ROOT

    monkeypatch.setenv("MIN_MARGIN", "1.5")
    with pytest.raises(ValueError):
        PipelineConfig.load(AI_ROOT / "project.json")
