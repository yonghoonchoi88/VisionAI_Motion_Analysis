"""FastAPI 계약 테스트: /health, /predict/video (가짜 YOLO + 가짜 RF)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import create_app


@pytest.fixture
def client_factory(make_pipeline, synthetic_bundle):
    def _make(schedule=lambda i: {1: 150, 2: 450}, fail: bool = False):
        def loader():
            if fail:
                raise FileNotFoundError("models/yolo11n-pose.pt")
            return synthetic_bundle, make_pipeline(schedule)
        return TestClient(create_app(loader))
    return _make


def _post(client, path, name="clip.mp4"):
    with open(path, "rb") as f:
        return client.post("/predict/video", files={"file": (name, f, "video/mp4")})


def test_health_ready(client_factory):
    with client_factory() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] is True
    assert body["model"]["n_features"] == 1140


def test_health_degraded_and_503(client_factory, make_video):
    with client_factory(fail=True) as c:
        body = c.get("/health").json()
        assert body["status"] == "degraded" and "yolo11n-pose.pt" in body["error"]
        r = _post(c, make_video(seconds=1))
    assert r.status_code == 503 and r.json()["detail"]["code"] == "MODEL_NOT_READY"


def test_predict_two_people(client_factory, make_video):
    """T02 + T04: 정상 영상 → 사람 2명의 예측이 track_id별로 반환."""
    with client_factory() as c:
        r = _post(c, make_video(seconds=6))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "OK" and body["person_count"] == 2
    assert [p["track_id"] for p in body["predictions"]] == [1, 2]
    p = body["predictions"][0]
    assert p["action_id"].startswith("A0") and 0 <= p["confidence"] <= 1
    assert body["video"]["processed_frames"] == 180
    assert body["processing_time_sec"] >= 0


def test_predict_no_person(client_factory, make_video):
    with client_factory(schedule=lambda i: {}) as c:
        body = _post(c, make_video(seconds=1)).json()
    assert body["status"] == "NO_PERSON" and body["predictions"] == []
    assert body["message"] == "영상에서 사람을 찾지 못했습니다."


def test_predict_corrupt_video(client_factory, corrupt_video):
    """T09: 손상 파일 → 422 + 사유 (Worker가 FAILURE로 저장)."""
    with client_factory() as c:
        r = _post(c, corrupt_video)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "VIDEO_UNREADABLE"


def test_predict_wrong_extension(client_factory, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with client_factory() as c:
        r = _post(c, f, name="notes.txt")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "UNSUPPORTED_FORMAT"


def test_predict_empty_file(client_factory, tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with client_factory() as c:
        r = _post(c, f)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "EMPTY_FILE"


def test_predict_reports_margin_and_rule(client_factory, make_video):
    """응답에 1·2등 차이(margin)·2등 동작·적용한 판단 기준이 포함된다 (Worker가 DB에 저장)."""
    with client_factory() as c:
        body = _post(c, make_video(seconds=2)).json()
        health = c.get("/health").json()
    p = body["predictions"][0]
    assert 0 <= p["margin"] <= p["confidence"]
    assert p["runner_up_label"] in p["probabilities"] and p["runner_up_label"] != p["action_label"]
    assert set(body["decision_rule"]) == {"min_confidence", "min_margin"}
    assert health["decision_rule"] == body["decision_rule"]


def test_v1_alias_paths_share_contract(client_factory, make_video):
    """개발 가이드 권장 경로 /api/v1/health · /api/v1/predict/ 도 같은 계약으로 응답한다."""
    with client_factory() as c:
        assert c.get("/api/v1/health").json() == c.get("/health").json()
        with open(make_video(seconds=2), "rb") as f:
            r = c.post("/api/v1/predict/", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200 and r.json()["status"] == "OK"
