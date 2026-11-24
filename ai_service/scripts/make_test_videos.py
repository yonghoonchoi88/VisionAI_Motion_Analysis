"""
수동 검증용 '비정상 입력' 영상 만들기 (개발 가이드 2단계: 사람이 없는 영상·너무 짧은 영상의 응답 확인)

만드는 파일 (기본 위치: ai_service/samples/ — *.mp4는 .gitignore로 Git 제외)
    no_person.mp4   3초 회색 화면                 → 200, status "NO_PERSON", predictions []
    broken.mp4      텍스트를 .mp4로 저장한 파일    → 422, detail.code "VIDEO_UNREADABLE"
    notes.txt       영상이 아닌 확장자             → 400, detail.code "UNSUPPORTED_FORMAT"
    too_short.mp4   (--from 영상을 줄 때만) 앞 0.3초만 자른 영상
                    → 사람이 10 Frame 미만으로 보임 → 200, status "NO_VALID_TRACK" + skipped_tracks 사유

실행 (ai_service/ 폴더에서)
    python -m scripts.make_test_videos --from ~/ai_vision/subject10_pose_action/data/day05/videos/S02/S02_A001_T01.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parent.parent


def write_gray(path: Path, seconds: float, fps: int = 25, size: tuple[int, int] = (640, 480)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        writer.write(np.full((size[1], size[0], 3), 90 + i % 20, dtype=np.uint8))
    writer.release()
    return path


def cut_head(src: Path, dst: Path, seconds: float) -> Path:
    """원본 영상의 앞부분만 잘라 저장한다 (사람이 너무 짧게 보이는 경우를 만들기 위함)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 영상을 열 수 없습니다: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(max(1, int(seconds * fps))):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="수동 검증용 비정상 입력 파일 생성")
    ap.add_argument("--out", type=Path, default=AI_ROOT / "samples")
    ap.add_argument("--from", dest="source", type=Path, help="too_short.mp4를 만들 원본 동작 영상 (선택)")
    ap.add_argument("--short-seconds", type=float, default=0.3)
    args = ap.parse_args(argv)

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    made = [write_gray(out / "no_person.mp4", 3.0)]
    (out / "broken.mp4").write_bytes(b"this is not a video\n" * 200)
    (out / "notes.txt").write_text("not a video\n", encoding="utf-8")
    made += [out / "broken.mp4", out / "notes.txt"]
    if args.source:
        made.append(cut_head(args.source.expanduser(), out / "too_short.mp4", args.short_seconds))

    for p in made:
        print(f"[OK] {p}  ({p.stat().st_size:,} bytes)")
    if not args.source:
        print("[참고] --from <동작 영상>을 주면 too_short.mp4(NO_VALID_TRACK 확인용)도 만듭니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
