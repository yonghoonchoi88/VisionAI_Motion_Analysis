"""Held-out 평가 스크립트의 채점·선택 규칙 (실제 YOLO 없이)."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate_holdout import (
    Sample,
    _run_video,
    choose_rule,
    choose_strategy,
    evaluate_rule,
    parse_video_name,
)
from services.action_inference import DecisionRule


def _s(label, pred, conf, margin, subject="S02", strategy="auto", main=True, video="v.mp4"):
    return Sample(video, subject, label, strategy, 1, main, pred, conf, margin, 30)


def test_parse_video_name():
    assert parse_video_name(Path("S02_A003_T01.mp4")) == ("S02", "walk_turn_walk")
    assert parse_video_name(Path("data/S03/s03_a006_t02.MP4")) == ("S03", "wave")
    assert parse_video_name(Path("01_sbu_shaking_hands.mp4")) is None
    assert parse_video_name(Path("S02_A999_T01.mp4")) is None


def test_evaluate_rule_counts_only_main_person_and_negatives():
    samples = [
        _s("wave", "wave", 0.35, 0.15),                         # 확정 정답
        _s("wave", "wave", 0.30, 0.15, main=False),             # 뒤에 지나간 사람 → 채점 제외
        _s("sit_stand", "wave", 0.22, 0.01),                    # 보류
        _s("", "drink_return", 0.25, 0.07, subject="NEG"),      # 미학습 동작을 확정 → 오답
    ]
    r = evaluate_rule(samples, DecisionRule(0.2, 0.05))
    assert (r["total"], r["top1"], r["confirmed"], r["confirmed_correct"], r["uncertain"]) == (2, 1, 1, 1, 1)
    assert (r["neg_total"], r["neg_confirmed"], r["confirmed_wrong"]) == (1, 1, 1)


def test_choose_rule_rejects_wrong_and_negatives():
    """정답은 margin이 크고, 오답·미학습 동작은 margin이 작은 표본 → 둘 사이에서 기준을 잡아야 한다."""
    samples = [
        _s("wave", "wave", 0.33, 0.14),
        _s("bend_return", "bend_return", 0.30, 0.11),
        _s("sit_stand", "wave", 0.22, 0.012),
        _s("", "leg_raise_lower", 0.25, 0.073, subject="NEG"),
    ]
    rule, r, n_ties = choose_rule(samples)
    assert r["confirmed_correct"] == 2 and r["confirmed_wrong"] == 0
    assert 0.073 <= rule.min_margin <= 0.11
    assert n_ties >= 1


def test_choose_strategy_prefers_current_on_tie():
    samples = [
        _s("wave", "wave", 0.3, 0.1, strategy="auto"),
        _s("wave", "wave", 0.3, 0.1, strategy="hybrid"),
        _s("wave", "bend_return", 0.3, 0.1, strategy="sliding"),
    ]
    best, top1 = choose_strategy(samples, ["hybrid", "auto", "sliding"], current="auto")
    assert best == "auto" and top1 == {"hybrid": 1, "auto": 1, "sliding": 0}


def test_run_video_marks_longest_track_as_main(make_pipeline, synthetic_bundle, make_video):
    """두 사람 중 오래 보인 사람이 채점 대상(주인공)이 된다."""
    pipe = make_pipeline(lambda i: {1: 150, 2: 450} if i < 20 else {1: 150})
    video = make_video(seconds=2, name="S02_A006_T01.mp4")
    samples = _run_video(pipe, synthetic_bundle, video, "S02", "wave", ["auto", "hybrid"])
    mains = [s for s in samples if s.is_main]
    assert {s.strategy for s in mains} == {"auto", "hybrid"}
    assert all(s.track_id == 1 for s in mains)
    assert all(0 <= s.margin <= s.confidence for s in samples)
