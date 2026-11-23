"""
IP Webcam 단시간 수집 (camera_capture.py)

Celery Worker가 호출한다. 허용된 카메라에 연결해 정해진 시간(기본 8초)만 Frame을 읽고,
그 구간을 MP4 1개로 저장한 뒤 연결을 끊는다. 이후 처리는 업로드 영상과 똑같다.

    capture_clip(url, seconds, out_path) → CaptureResult

- '8초'는 입력 영상의 길이. 연결·AI 추론 시간은 별도.
- FPS는 실제로 받은 Frame 수 / 실제 경과 시간으로 계산해 저장 → 영상 속 시간이 실제와 같아진다.
- 연결 실패·도중 끊김·Frame 부족은 CameraCaptureError (사용자에게 그대로 보여 줄 메시지).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_MS = 5000          # 카메라 연결 대기 한도
READ_TIMEOUT_MS = 5000          # Frame 1장 대기 한도
MAX_CONSECUTIVE_FAILS = 30      # 연속 읽기 실패 허용 횟수 (넘으면 연결 끊김으로 판단)
MIN_FRAMES = 15                 # 이보다 적게 받으면 분석 불가
MAX_WIDTH = 960                 # 큰 해상도는 줄여서 저장 (YOLO 입력은 640)


class CameraCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    frames: int
    seconds: float
    fps: float
    width: int
    height: int


def capture_clip(source: str, seconds: float, out_path: Path | str) -> CaptureResult:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(
        source,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS],
    )
    if not cap.isOpened():
        raise CameraCaptureError("카메라에 연결할 수 없습니다. 휴대폰 IP Webcam 앱이 켜져 있고 같은 네트워크인지 확인하세요.")

    # 메모리 절약을 위해 JPEG로 압축해 모아 두었다가, 실제 FPS를 안 뒤 한 번에 MP4로 쓴다.
    encoded: list[bytes] = []
    size: tuple[int, int] | None = None
    fails = 0
    try:
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= MAX_CONSECUTIVE_FAILS:
                    raise CameraCaptureError(f"수집 중 카메라 연결이 끊겼습니다 ({elapsed:.1f}초 시점).")
                time.sleep(0.02)
                continue
            fails = 0
            frame = _limit_width(frame)
            size = size or (frame.shape[1], frame.shape[0])
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                encoded.append(buf.tobytes())
        actual = time.monotonic() - start
    finally:
        cap.release()

    if len(encoded) < MIN_FRAMES or size is None:
        raise CameraCaptureError(f"받은 Frame이 너무 적습니다 ({len(encoded)}장). 카메라 상태를 확인하세요.")

    # MPEG-4는 timebase 분모가 65535 이하여야 한다 → 측정 FPS(예: 14.837…)를 소수 2자리로 반올림
    fps = max(1.0, round(len(encoded) / actual, 2))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():                      # 그래도 실패하면 정수 FPS로 재시도
        fps = float(max(1, round(fps)))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise CameraCaptureError("수집한 영상을 저장하지 못했습니다.")
    try:
        for data in encoded:
            writer.write(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    finally:
        writer.release()

    logger.info("captured %d frames in %.2fs (%.1f fps) → %s", len(encoded), actual, fps, out_path)
    return CaptureResult(out_path, len(encoded), round(actual, 2), round(fps, 2), size[0], size[1])


def _limit_width(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= MAX_WIDTH:
        return frame
    scale = MAX_WIDTH / w
    return cv2.resize(frame, (MAX_WIDTH, int(h * scale)))
