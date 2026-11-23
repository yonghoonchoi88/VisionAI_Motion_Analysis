"""
테스트 공통 준비물 — GPU·YOLO·대용량 모델 없이 전체 흐름을 검증한다.

- synthetic_bundle : models/의 '진짜' JSON 2개 + 같은 규격의 작은 가짜 Random Forest
- FakePoseModel    : YOLO.track()과 같은 모양의 결과를 돌려주는 가짜 Pose 모델
- make_video       : OpenCV로 만든 짧은 테스트 영상
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from services.model_loader import (
    FEATURE_COLUMNS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    TRAINING_INFO_FILE,
    load_model_bundle,
)
from services.pose_pipeline import PipelineConfig, PosePipeline

AI_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────
# 모델 번들
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def synthetic_models_dir(tmp_path_factory) -> Path:
    """진짜 JSON 규격 + 가짜 RF. (joblib은 Git에 없으므로 CI에서도 돌아가게)"""
    d = tmp_path_factory.mktemp("models")
    shutil.copy(MODELS_DIR / FEATURE_COLUMNS_FILE, d)
    shutil.copy(MODELS_DIR / TRAINING_INFO_FILE, d)
    info = json.loads((d / TRAINING_INFO_FILE).read_text(encoding="utf-8"))

    rng = np.random.default_rng(0)
    classes = sorted(info["classes"])
    x = rng.normal(size=(len(classes) * 10, info["feature_count"]))
    y = np.repeat(classes, 10)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    joblib.dump(model, d / MODEL_FILE)
    return d


@pytest.fixture(scope="session")
def synthetic_bundle(synthetic_models_dir):
    return load_model_bundle(synthetic_models_dir)


# ─────────────────────────────────────────────────────────────
# 가짜 YOLO Pose
# ─────────────────────────────────────────────────────────────
def standing_pose(offset_x: float, phase: float) -> np.ndarray:
    """정면으로 선 사람의 COCO 17 Keypoint (px). phase로 오른팔을 흔든다."""
    kp = np.array([
        [0, -160], [-8, -168], [8, -168], [-16, -164], [16, -164],   # 얼굴
        [-40, -120], [40, -120],                                      # 어깨
        [-55, -70], [55, -70],                                        # 팔꿈치
        [-60, -20], [60, -20],                                        # 손목
        [-25, 0], [25, 0],                                            # 골반
        [-27, 80], [27, 80],                                          # 무릎
        [-28, 160], [28, 160],                                        # 발목
    ], dtype=np.float32)
    kp[10] = [60 + 40 * np.cos(phase), -150 + 30 * np.sin(phase)]     # 오른손 흔들기
    kp[:, 0] += offset_x
    kp[:, 1] += 240
    return kp


class FakeTracker:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class FakePoseModel:
    """schedule(frame_index) → {track_id: x_offset} 로 어떤 사람이 언제 보일지 정한다."""

    def __init__(self, schedule):
        self.schedule = schedule
        self.frame_index = 0
        self.predictor = SimpleNamespace(trackers=[FakeTracker()])
        self.calls = 0

    def track(self, source, **kwargs):
        self.calls += 1
        people = self.schedule(self.frame_index)
        self.frame_index += 1
        if not people:
            return [SimpleNamespace(boxes=SimpleNamespace(id=None), keypoints=SimpleNamespace(xy=np.zeros((0, 17, 2))))]
        ids = np.array(list(people.keys()), dtype=np.float32)
        kps = np.stack([standing_pose(x, self.frame_index / 5) for x in people.values()])
        return [SimpleNamespace(boxes=SimpleNamespace(id=ids), keypoints=SimpleNamespace(xy=kps))]


@pytest.fixture
def config() -> PipelineConfig:
    cfg = PipelineConfig.load(AI_ROOT / "project.json")
    return PipelineConfig(**{**cfg.__dict__, "device": "cpu"})


@pytest.fixture
def make_pipeline(config, synthetic_bundle):
    def _make(schedule, **overrides):
        cfg = PipelineConfig(**{**config.__dict__, **overrides})
        return PosePipeline(
            cfg,
            sequence_steps=synthetic_bundle.sequence_steps,
            min_valid_frames=synthetic_bundle.min_valid_frames,
            pose_model=FakePoseModel(schedule),
        )
    return _make


# ─────────────────────────────────────────────────────────────
# 테스트 영상
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def make_video(tmp_path):
    def _make(seconds: float, fps: int = 30, name: str = "clip.mp4") -> Path:
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
        for i in range(int(seconds * fps)):
            frame = np.full((120, 160, 3), i % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path
    return _make


@pytest.fixture
def corrupt_video(tmp_path) -> Path:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video" * 100)
    return path
