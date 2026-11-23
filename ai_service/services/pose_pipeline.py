"""
영상 → 사람별 30-Step Sequence (pose_pipeline.py)

역할
    영상 파일 1개를 읽어 YOLO Pose + ByteTrack으로 사람을 추적하고,
    사람(track_id)마다 모델 입력용 Window 행렬 (n_windows, 1140)을 만든다.

원본 재사용 (학습 때와 같은 숫자가 나와야 하므로 계산 코드는 새로 쓰지 않는다)
    - src.multi_person_utils.feature_from_keypoints  : 17 Keypoint → 38-D Feature (정규화·각도)
    - src.multi_person_utils.MultiPersonStateManager : Track ID별 5초 Buffer·Lost Timeout 관리
    - src.sequence_utils.resample_sequence / flatten_sequence : Window → 30 Step → 1140
    흐름은 8일차 scripts/49_ip_webcam_multi_person_action.py의 Frame 루프와 같다.
    차이점: 실시간 카메라 대신 '파일'을 읽고, 시간은 벽시계 대신 frame_index / fps를 쓴다.

Window 생성 방식 (project.json window.strategy, 환경변수 WINDOW_STRATEGY로 덮어쓰기)
    sliding     : 7일차 규칙 — Buffer가 준비(is_ready)되면 0.25초마다 최근 5초로 Window 1개 생성
                  (Window가 하나도 없으면 whole_track으로 대체)
    whole_track : 사람마다 보인 전체 구간을 30 Step으로 리샘플한 Window 1개 (6일차 학습 Clip 방식과 동일)
    auto        : 사람이 보인 시간이 whole_track_max_seconds 이하면 whole_track, 길면 sliding
    공통        : 유효 Frame이 min_valid_frames(10) 미만인 사람은 skipped_tracks로 보고
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.multi_person_utils import MultiPersonStateManager, feature_from_keypoints
from src.sequence_utils import flatten_sequence, resample_sequence

logger = logging.getLogger(__name__)

AI_ROOT = Path(__file__).resolve().parent.parent          # ai_service/
CONFIG_PATH = AI_ROOT / "project.json"
DEFAULT_FPS = 30.0                                        # 영상에 FPS 정보가 없을 때 가정값
VALID_STRATEGIES = ("sliding", "whole_track", "auto", "hybrid")


class VideoReadError(ValueError):
    """영상 파일을 열 수 없거나, 손상됐거나, 허용 길이를 넘음 → API에서 422로 응답.

    code: "VIDEO_UNREADABLE" | "VIDEO_TOO_LONG" (API 오류 코드로 그대로 사용)
    """

    def __init__(self, message: str, code: str = "VIDEO_UNREADABLE"):
        super().__init__(message)
        self.code = code


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    pose_model_path: Path
    image_size: int
    person_confidence: float
    device: str
    tracker: str
    lost_timeout_seconds: float
    window_strategy: str
    whole_track_max_seconds: float
    window_seconds: float
    min_window_fill_ratio: float
    min_valid_feature_ratio: float
    min_valid_features: int
    prediction_interval_sec: float
    min_confidence: float          # 판단 불확실 기준 ①: 1등 확률이 이보다 낮으면 불확실
    min_margin: float              # 판단 불확실 기준 ②: 1등-2등 확률 차가 이보다 작으면 불확실
    max_video_seconds: float

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "PipelineConfig":
        """project.json을 읽는다.

        환경변수로 덮어쓰기: POSE_DEVICE(예: Docker에서 cpu) · WINDOW_STRATEGY · MIN_CONFIDENCE · MIN_MARGIN
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        pose, win, trk, lim = raw["pose"], raw["window"], raw["tracking"], raw["limits"]

        strategy = os.getenv("WINDOW_STRATEGY", str(win.get("strategy", "sliding")))
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"window.strategy는 {VALID_STRATEGIES} 중 하나여야 합니다: {strategy}")

        # 판단 기준: decision 블록이 없으면 예전 설정(window.confidence_threshold, margin 0)으로 동작
        decision = raw.get("decision", {})
        min_confidence = float(os.getenv("MIN_CONFIDENCE", decision.get("min_confidence", win.get("confidence_threshold", 0.55))))
        min_margin = float(os.getenv("MIN_MARGIN", decision.get("min_margin", 0.0)))
        if not (0.0 <= min_confidence <= 1.0 and 0.0 <= min_margin <= 1.0):
            raise ValueError(f"판단 기준은 0~1 사이여야 합니다: min_confidence={min_confidence}, min_margin={min_margin}")

        model_path = Path(pose["model"])
        if not model_path.is_absolute():
            model_path = path.parent / model_path

        return cls(
            pose_model_path=model_path,
            image_size=int(pose["image_size"]),
            person_confidence=float(pose["person_confidence"]),
            device=os.getenv("POSE_DEVICE", str(pose["device"])),
            tracker=str(trk["tracker"]),
            lost_timeout_seconds=float(trk["lost_timeout_seconds"]),
            window_strategy=strategy,
            whole_track_max_seconds=float(win.get("whole_track_max_seconds", 12.0)),
            window_seconds=float(win["window_seconds"]),
            min_window_fill_ratio=float(win["min_window_fill_ratio"]),
            min_valid_feature_ratio=float(win["min_valid_feature_ratio"]),
            min_valid_features=int(win["min_valid_features"]),
            prediction_interval_sec=float(win["prediction_interval_sec"]),
            min_confidence=min_confidence,
            min_margin=min_margin,
            max_video_seconds=float(os.getenv("MAX_VIDEO_SECONDS", lim["max_video_seconds"])),
        )


def resolve_device(requested: str) -> Any:
    """원본 src/common.py의 resolve_device와 같은 규칙: auto면 CUDA가 있을 때 GPU 0, 없으면 cpu."""
    if requested != "auto":
        return requested
    import torch  # 무거운 import는 필요할 때만

    return 0 if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# 결과 구조
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VideoInfo:
    fps: float
    fps_assumed: bool          # 영상에 FPS 정보가 없어 30으로 가정했는지
    frame_count: int           # 컨테이너가 알려 준 전체 Frame 수 (0이면 미상)
    processed_frames: int      # 실제로 읽고 처리한 Frame 수
    duration_sec: float        # processed_frames / fps
    width: int
    height: int


@dataclass(frozen=True)
class TrackSequences:
    track_id: int
    windows: np.ndarray        # (n_windows, 1140). hybrid면 0번 행 = 전체 구간, 1번~ = sliding Window
    window_mode: str           # "sliding" | "whole_track" | "hybrid"
    seen_frames: int           # 이 사람이 검출된 Frame 수
    valid_frames: int          # 그중 38-D Feature 계산에 성공한 Frame 수
    first_sec: float
    last_sec: float

    @property
    def duration_sec(self) -> float:
        return round(self.last_sec - self.first_sec, 3)

    @property
    def valid_frame_ratio(self) -> float:
        return round(self.valid_frames / self.seen_frames, 4) if self.seen_frames else 0.0


@dataclass(frozen=True)
class SkippedTrack:
    track_id: int
    seen_frames: int
    valid_frames: int
    reason: str


@dataclass(frozen=True)
class PipelineOutput:
    video: VideoInfo
    tracks: list[TrackSequences] = field(default_factory=list)
    skipped_tracks: list[SkippedTrack] = field(default_factory=list)

    @property
    def detected_person_count(self) -> int:
        return len(self.tracks) + len(self.skipped_tracks)


@dataclass
class _TrackLog:
    """파일 전체에 걸친 사람별 누적 기록 (Buffer가 Lost Timeout으로 지워져도 남는다)."""
    windows: list[np.ndarray] = field(default_factory=list)
    valid_times: list[float] = field(default_factory=list)
    valid_features: list[np.ndarray] = field(default_factory=list)
    seen_frames: int = 0
    first_sec: float = math.inf
    last_sec: float = -math.inf


def _to_numpy(value: Any) -> np.ndarray:
    """torch Tensor(GPU 포함)·NumPy 모두 NumPy로."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


# ─────────────────────────────────────────────────────────────
# 파이프라인
# ─────────────────────────────────────────────────────────────
class PosePipeline:
    """YOLO Pose 모델을 1회 로드해 두고, 영상마다 run()을 호출한다.

    Tracker 상태는 모델 객체에 붙어 있으므로 동시에 두 영상을 처리하면 섞인다.
    → 내부 Lock으로 한 번에 한 영상만 처리한다. (Celery Worker가 순차 호출하는 구조와도 맞음)
    """

    def __init__(
        self,
        config: PipelineConfig,
        sequence_steps: int,
        min_valid_frames: int,
        pose_model: Any | None = None,       # 테스트에서 가짜 모델을 주입할 때 사용
    ):
        self.config = config
        self.sequence_steps = int(sequence_steps)
        self.min_valid_frames = int(min_valid_frames)
        self.device = resolve_device(config.device)
        self._lock = threading.Lock()

        if pose_model is None:
            if not config.pose_model_path.is_file():
                raise FileNotFoundError(f"Pose 가중치가 없습니다: {config.pose_model_path}")
            from ultralytics import YOLO

            pose_model = YOLO(str(config.pose_model_path))
        self.model = pose_model

    # ── 공개 API ────────────────────────────────────────────
    def warmup(self) -> None:
        """서버 시작 시 빈 Frame으로 1회 추론 → 첫 요청이 느려지는 것(CUDA 초기화 등)을 막는다."""
        with self._lock:
            blank = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
            self._track(blank)
            self._reset_trackers()

    def run(self, video_path: Path | str) -> PipelineOutput:
        with self._lock:
            video, logs = self._collect(Path(video_path))
        return self._finalize(video, logs, self.config.window_strategy)

    def run_strategies(self, video_path: Path | str, strategies: list[str] | tuple[str, ...]) -> dict[str, PipelineOutput]:
        """YOLO는 영상당 1번만 돌리고, 같은 Pose 기록으로 여러 Window 전략의 결과를 만든다.

        전략 A/B 비교(scripts/evaluate_holdout.py)에서 같은 입력으로 공정하게 비교하기 위해 사용한다.
        """
        unknown = [s for s in strategies if s not in VALID_STRATEGIES]
        if unknown:
            raise ValueError(f"알 수 없는 전략: {unknown} (가능: {VALID_STRATEGIES})")
        with self._lock:
            video, logs = self._collect(Path(video_path))
        return {s: self._finalize(video, logs, s) for s in strategies}

    # ── 내부 구현 ───────────────────────────────────────────
    def _track(self, frame: np.ndarray):
        return self.model.track(
            source=frame,
            persist=True,                         # 같은 영상의 Frame 사이 Tracker 상태 유지
            tracker=self.config.tracker,
            imgsz=self.config.image_size,
            conf=self.config.person_confidence,
            device=self.device,
            verbose=False,
        )[0]

    def _reset_trackers(self) -> None:
        """새 영상마다 ByteTrack 상태와 Track ID(1부터)를 초기화한다. 안 하면 이전 요청의 사람과 섞인다."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def _collect(self, video_path: Path) -> tuple[VideoInfo, dict[int, "_TrackLog"]]:
        """영상 전체 Frame에 YOLO Pose·ByteTrack을 돌려 사람별 기록(_TrackLog)을 모은다."""
        cfg = self.config
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoReadError("영상을 열 수 없습니다. 손상되었거나 지원하지 않는 코덱입니다.")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps_assumed = not (math.isfinite(fps) and 1.0 <= fps <= 240.0)
            if fps_assumed:
                fps = DEFAULT_FPS
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if frame_count > 0 and frame_count / fps > cfg.max_video_seconds:
                raise VideoReadError(
                    f"영상이 너무 깁니다: {frame_count / fps:.1f}초 (최대 {cfg.max_video_seconds:.0f}초)",
                    code="VIDEO_TOO_LONG",
                )

            self._reset_trackers()
            manager = MultiPersonStateManager(
                window_seconds=cfg.window_seconds,
                stabilization_buffer=1,            # 안정화 투표는 쓰지 않음 (확률 평균으로 집계)
                stabilization_min_votes=1,
                lost_timeout_seconds=cfg.lost_timeout_seconds,
            )
            logs: dict[int, _TrackLog] = {}
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                now = frame_index / fps
                if now > cfg.max_video_seconds:   # 메타데이터가 틀린 영상 대비 2차 안전장치
                    break
                self._process_frame(frame, now, manager, logs)
                frame_index += 1
        finally:
            cap.release()

        if frame_index == 0:
            raise VideoReadError("영상에서 Frame을 하나도 읽지 못했습니다. 파일이 손상되었을 수 있습니다.")

        video = VideoInfo(
            fps=round(fps, 3),
            fps_assumed=fps_assumed,
            frame_count=frame_count,
            processed_frames=frame_index,
            duration_sec=round(frame_index / fps, 3),
            width=width,
            height=height,
        )
        return video, logs

    def _process_frame(
        self,
        frame: np.ndarray,
        now: float,
        manager: MultiPersonStateManager,
        logs: dict[int, _TrackLog],
    ) -> None:
        """8일차 49번 스크립트의 Frame 루프 본문과 같은 순서로 처리한다."""
        cfg = self.config
        result = self._track(frame)
        boxes, keypoints = result.boxes, result.keypoints
        seen_ids: set[int] = set()

        if boxes is not None and boxes.id is not None and keypoints is not None and keypoints.xy is not None:
            track_ids = _to_numpy(boxes.id).astype(int)
            kp_xy = _to_numpy(keypoints.xy)

            for i in range(min(len(track_ids), len(kp_xy))):
                track_id = int(track_ids[i])
                seen_ids.add(track_id)
                state = manager.get_or_create(track_id, now)
                log = logs.setdefault(track_id, _TrackLog())

                feature = feature_from_keypoints(kp_xy[i])      # 원본 함수: None이면 이 Frame은 무효
                state.feature_buffer.append(now, feature)

                log.seen_frames += 1
                log.first_sec = min(log.first_sec, now)
                log.last_sec = max(log.last_sec, now)
                if feature is not None:
                    log.valid_times.append(now)
                    log.valid_features.append(feature)

                ready = state.feature_buffer.is_ready(
                    min_window_fill_ratio=cfg.min_window_fill_ratio,
                    min_valid_feature_ratio=cfg.min_valid_feature_ratio,
                    min_valid_features=cfg.min_valid_features,
                )
                if ready and (now - state.last_prediction_time) >= cfg.prediction_interval_sec:
                    matrix, times = state.feature_buffer.valid_matrix_and_times()
                    log.windows.append(self._to_window(matrix, times))
                    state.last_prediction_time = now

        manager.mark_missing(now, seen_ids)      # 이번 Frame에 없던 사람: Buffer에 None 기록
        manager.remove_lost(now)                 # Lost Timeout 지난 사람: Buffer 제거 (logs는 유지)

    def _to_window(self, matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
        """(T, 38) 가변 길이 → (30, 38) 리샘플 → (1140,). 원본 predict_from_window와 같은 계산."""
        sequence = resample_sequence(matrix, times, target_steps=self.sequence_steps)
        return flatten_sequence(sequence)

    def _choose_mode(self, log: _TrackLog, strategy: str) -> str:
        """전략에 따라 이 사람의 Window 방식을 고른다. sliding Window가 없으면 항상 whole_track."""
        if strategy == "auto":
            visible = log.last_sec - log.first_sec
            strategy = "whole_track" if visible <= self.config.whole_track_max_seconds else "sliding"
        if strategy in ("sliding", "hybrid") and log.windows:
            return strategy
        return "whole_track"

    def _finalize(self, video: VideoInfo, logs: dict[int, _TrackLog], strategy: str) -> PipelineOutput:
        tracks: list[TrackSequences] = []
        skipped: list[SkippedTrack] = []

        for track_id in sorted(logs):
            log = logs[track_id]
            valid = len(log.valid_features)

            if valid < self.min_valid_frames:
                skipped.append(
                    SkippedTrack(
                        track_id=track_id,
                        seen_frames=log.seen_frames,
                        valid_frames=valid,
                        reason=f"유효 Frame 부족 ({valid} < {self.min_valid_frames})",
                    )
                )
                continue

            mode = self._choose_mode(log, strategy)
            if mode == "sliding":
                windows = np.stack(log.windows)
            else:
                # 6일차 학습과 같은 입력: 이 사람이 보인 구간 전체를 30 Step으로 리샘플
                matrix = np.stack(log.valid_features).astype(np.float32)
                times = np.asarray(log.valid_times, dtype=np.float32)
                whole = self._to_window(matrix, times)[np.newaxis, :]
                # hybrid: [전체 구간, sliding Window...] → action_inference가 반반 가중 평균
                windows = np.vstack([whole, np.stack(log.windows)]) if mode == "hybrid" else whole

            tracks.append(
                TrackSequences(
                    track_id=track_id,
                    windows=windows,
                    window_mode=mode,
                    seen_frames=log.seen_frames,
                    valid_frames=valid,
                    first_sec=round(log.first_sec, 3),
                    last_sec=round(log.last_sec, 3),
                )
            )

        logger.info(
            "pipeline done: frames=%d tracks=%d skipped=%d",
            video.processed_frames, len(tracks), len(skipped),
        )
        return PipelineOutput(video=video, tracks=tracks, skipped_tracks=skipped)
