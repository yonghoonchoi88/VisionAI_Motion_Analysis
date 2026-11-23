"""영상 → 사람별 Window 생성 규칙 (가짜 YOLO 사용)."""

from __future__ import annotations

import numpy as np
import pytest

from services.pose_pipeline import VideoReadError


def test_two_people_get_separate_tracks(make_pipeline, make_video):
    """T04: 두 사람이 계속 보이면 Track ID별로 분리되고 sliding Window가 만들어진다."""
    pipe = make_pipeline(lambda i: {1: 150, 2: 450})
    out = pipe.run(make_video(seconds=6))

    assert [t.track_id for t in out.tracks] == [1, 2]
    for t in out.tracks:
        assert t.window_mode == "sliding"
        assert t.windows.shape[1] == 1140
        assert t.windows.shape[0] >= 2
        assert np.isfinite(t.windows).all()
        assert t.valid_frame_ratio == 1.0
    assert out.video.processed_frames == 180


def test_short_clip_uses_whole_track(make_pipeline, make_video):
    """5초 Window가 안 차는 짧은 영상도 유효 Frame ≥ 10이면 전체 구간 1개 Window로 예측한다."""
    out = make_pipeline(lambda i: {1: 300}).run(make_video(seconds=2))
    assert len(out.tracks) == 1
    assert out.tracks[0].window_mode == "whole_track"
    assert out.tracks[0].windows.shape == (1, 1140)


def test_brief_person_is_skipped(make_pipeline, make_video):
    """5 Frame만 스쳐간 사람은 예측하지 않고 skipped_tracks로 보고한다."""
    out = make_pipeline(lambda i: {1: 300, 9: 100} if i < 5 else {1: 300}).run(make_video(seconds=2))
    assert [t.track_id for t in out.tracks] == [1]
    assert [s.track_id for s in out.skipped_tracks] == [9]
    assert "유효 Frame 부족" in out.skipped_tracks[0].reason


def test_no_person(make_pipeline, make_video):
    out = make_pipeline(lambda i: {}).run(make_video(seconds=1))
    assert out.tracks == [] and out.skipped_tracks == []
    assert out.detected_person_count == 0


def test_corrupt_video_raises(make_pipeline, corrupt_video):
    with pytest.raises(VideoReadError) as e:
        make_pipeline(lambda i: {1: 300}).run(corrupt_video)
    assert e.value.code == "VIDEO_UNREADABLE"


def test_too_long_video_raises(make_pipeline, make_video):
    with pytest.raises(VideoReadError) as e:
        make_pipeline(lambda i: {1: 300}, max_video_seconds=1.0).run(make_video(seconds=2))
    assert e.value.code == "VIDEO_TOO_LONG"


def test_tracker_reset_every_video(make_pipeline, make_video):
    """영상마다 Tracker를 초기화해야 이전 요청의 사람 ID와 섞이지 않는다."""
    pipe = make_pipeline(lambda i: {1: 300})
    video = make_video(seconds=1)
    pipe.run(video)
    pipe.run(video)
    assert pipe.model.predictor.trackers[0].reset_count == 2


@pytest.mark.parametrize(
    ("strategy", "expected_mode", "expected_windows"),
    [("sliding", "sliding", None), ("whole_track", "whole_track", 1), ("auto", "whole_track", 1)],
)
def test_window_strategy(make_pipeline, make_video, strategy, expected_mode, expected_windows):
    """10초 영상: sliding은 Window 여러 개, whole_track·auto(≤12초)는 전체 구간 1개."""
    out = make_pipeline(lambda i: {1: 300}, window_strategy=strategy).run(make_video(seconds=10))
    t = out.tracks[0]
    assert t.window_mode == expected_mode
    if expected_windows is None:
        assert t.windows.shape[0] > 1
    else:
        assert t.windows.shape == (expected_windows, 1140)


def test_auto_uses_sliding_for_long_track(make_pipeline, make_video):
    out = make_pipeline(lambda i: {1: 300}, window_strategy="auto", whole_track_max_seconds=3.0).run(make_video(seconds=6))
    assert out.tracks[0].window_mode == "sliding"
