"""T01 모델 로드·규격 검증 + 사람별 추론 집계."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from services.action_inference import predict_tracks
from services.model_loader import (
    FEATURE_COLUMNS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    ModelSpecError,
    load_model_bundle,
)


@pytest.mark.skipif(not (MODELS_DIR / MODEL_FILE).is_file(), reason="실제 joblib 모델이 없는 환경 (Git 미포함)")
def test_t01_real_model_loads():
    bundle = load_model_bundle()
    assert bundle.n_features == 1140
    assert bundle.sequence_steps * bundle.features_per_step == 1140
    assert len(bundle.classes) == 6


def test_synthetic_bundle_matches_spec(synthetic_bundle):
    assert synthetic_bundle.n_features == 1140
    assert list(synthetic_bundle.classes) == sorted(synthetic_bundle.classes)  # sklearn은 알파벳순
    assert synthetic_bundle.action_id("wave") == "A006"
    assert synthetic_bundle.action_id("drink_return") == "A004"


def test_missing_files_reported(tmp_path):
    with pytest.raises(ModelSpecError, match="모델 산출물이 없습니다"):
        load_model_bundle(tmp_path)


def test_broken_feature_columns_detected(tmp_path, synthetic_models_dir):
    for f in synthetic_models_dir.iterdir():
        shutil.copy(f, tmp_path)
    cols = json.loads((tmp_path / FEATURE_COLUMNS_FILE).read_text(encoding="utf-8"))
    cols[0], cols[1] = cols[1], cols[0]                        # 순서만 바꿔도 잡아야 함
    (tmp_path / FEATURE_COLUMNS_FILE).write_text(json.dumps(cols), encoding="utf-8")
    with pytest.raises(ModelSpecError):
        load_model_bundle(tmp_path)


def test_predict_tracks_sorted_and_named(synthetic_bundle):
    rng = np.random.default_rng(1)
    preds = predict_tracks(synthetic_bundle, {7: rng.normal(size=(3, 1140)), 2: rng.normal(size=(2, 30, 38))})
    assert [p.track_id for p in preds] == [2, 7]
    for p in preds:
        assert p.action_label in synthetic_bundle.classes
        assert p.action_id == synthetic_bundle.action_id(p.action_label)
        assert abs(sum(p.probabilities.values()) - 1) < 1e-3
        assert p.confidence == max(p.probabilities.values())


def test_predict_tracks_rejects_bad_input(synthetic_bundle):
    with pytest.raises(ValueError, match="1140"):
        predict_tracks(synthetic_bundle, {1: np.zeros((2, 100))})
    with pytest.raises(ValueError, match="NaN"):
        predict_tracks(synthetic_bundle, {1: np.full((1, 1140), np.nan)})
    assert predict_tracks(synthetic_bundle, {}) == []
