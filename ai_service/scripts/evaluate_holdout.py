"""
Held-out 평가 — 학습에 쓰지 않은 사람의 영상으로 서비스 추론을 검증한다.

왜 필요한가
    6일차 모델은 S01(1명)의 영상 6개로 학습했다. 학습 영상으로 확인하면 정답률이 높게 나오는 것이 당연하므로
    "서비스가 새 사람에게도 동작하는가"는 S02·S03처럼 학습에 쓰지 않은 사람의 영상으로만 말할 수 있다.

규칙 (테스트 데이터로 튜닝하지 않기)
    1) S02(검증용 사람)로만 Window 전략과 판단 기준(min_confidence, min_margin)을 고른다.
    2) S03(테스트용 사람)은 고른 설정을 '그대로' 한 번 적용해 결과만 보고한다.
    3) --negatives로 학습하지 않은 동작 영상(예: SBU 악수)을 주면, 그 영상의 예측은 모두 '불확실'이어야 한다.

판단 기준 선택 방법
    격자(min_confidence 0.15~0.60, min_margin 0.00~0.30, 0.01 간격)마다
        점수 = 확정한 정답 수 − 2 × 확정한 오답 수   (오답을 확정하는 것이 보류보다 나쁘다는 서비스 원칙)
    최고 점수인 조합이 여러 개면 그 가운데(중앙값) 조합을 고른다 — 경계값에 딱 붙은 값은 표본이 적을 때 불안정하다.

실행 (ai_service/ 폴더에서, 실제 모델·YOLO 필요)
    python -m scripts.evaluate_holdout \\
        --videos ~/ai_vision/subject10_pose_action/data/day05/videos \\
        --negatives ~/ai_vision/subject10_pose_action/data/day08/videos      # 선택

결과
    reports/evidence/holdout/holdout_eval.md      요약 표 (README에 그대로 인용)
    reports/evidence/holdout/predictions.csv      영상 × 전략 × 사람별 원본 예측
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

AI_ROOT = Path(__file__).resolve().parent.parent
if str(AI_ROOT) not in sys.path:                     # `python scripts/evaluate_holdout.py`로 실행해도 import되게
    sys.path.insert(0, str(AI_ROOT))

from services.action_inference import DecisionRule, predict_tracks  # noqa: E402
from services.model_loader import ACTION_IDS  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STRATEGIES = ("auto", "whole_track", "sliding", "hybrid")
NAME_PATTERN = re.compile(r"(S\d{2})_(A\d{3})_T\d+", re.IGNORECASE)
LABEL_BY_ID = {aid: label for label, aid in ACTION_IDS.items()}
WRONG_PENALTY = 2
CONF_GRID = np.round(np.arange(0.15, 0.6001, 0.01), 2)
MARGIN_GRID = np.round(np.arange(0.0, 0.3001, 0.01), 2)


@dataclass(frozen=True)
class Sample:
    """평가 단위 1건 = (영상, 전략, 사람)."""
    video: str
    subject: str            # S02 / S03 / NEG
    true_label: str         # NEG면 "" (정답 없음 — 불확실이어야 함)
    strategy: str
    track_id: int
    is_main: bool           # 정답이 있는 영상에서 가장 오래 보인 사람 = 채점 대상
    pred_label: str
    confidence: float
    margin: float
    valid_frames: int

    @property
    def correct(self) -> bool:
        return bool(self.true_label) and self.pred_label == self.true_label


# ─────────────────────────────────────────────────────────────
# 순수 함수 (tests/test_evaluate_holdout.py에서 검증)
# ─────────────────────────────────────────────────────────────
def parse_video_name(path: Path) -> tuple[str, str] | None:
    """'S02_A003_T01.mp4' → ('S02', 'walk_turn_walk'). 규칙에 안 맞으면 None."""
    m = NAME_PATTERN.search(path.stem)
    if not m:
        return None
    label = LABEL_BY_ID.get(m.group(2).upper())
    return (m.group(1).upper(), label) if label else None


def scored(samples: list[Sample]) -> list[Sample]:
    """채점 대상: 정답이 있는 영상은 주인공 1명, 미학습 동작 영상은 모든 사람."""
    return [s for s in samples if s.is_main or s.subject == "NEG"]


def evaluate_rule(samples: list[Sample], rule: DecisionRule) -> dict[str, int]:
    """한 판단 기준을 적용했을 때의 확정/보류 집계."""
    out = {"total": 0, "top1": 0, "confirmed": 0, "confirmed_correct": 0, "confirmed_wrong": 0, "uncertain": 0,
           "neg_total": 0, "neg_confirmed": 0}
    for s in scored(samples):
        uncertain = rule.is_uncertain(s.confidence, s.margin)
        if s.subject == "NEG":
            out["neg_total"] += 1
            out["neg_confirmed"] += int(not uncertain)
            out["confirmed_wrong"] += int(not uncertain)
            continue
        out["total"] += 1
        out["top1"] += int(s.correct)
        if uncertain:
            out["uncertain"] += 1
        else:
            out["confirmed"] += 1
            out["confirmed_correct"] += int(s.correct)
            out["confirmed_wrong"] += int(not s.correct)
    return out


def choose_rule(samples: list[Sample]) -> tuple[DecisionRule, dict[str, int], int]:
    """격자 탐색 → (고른 기준, 그 기준의 집계, 최고 점수 조합 수)."""
    best_score, best = None, []
    for c in CONF_GRID:
        for m in MARGIN_GRID:
            rule = DecisionRule(float(c), float(m))
            r = evaluate_rule(samples, rule)
            score = r["confirmed_correct"] - WRONG_PENALTY * r["confirmed_wrong"]
            if best_score is None or score > best_score:
                best_score, best = score, [rule]
            elif score == best_score:
                best.append(rule)
    best.sort(key=lambda r: (r.min_margin, r.min_confidence))
    chosen = best[len(best) // 2]
    return chosen, evaluate_rule(samples, chosen), len(best)


def choose_strategy(samples: list[Sample], strategies: list[str], current: str) -> tuple[str, dict[str, int]]:
    """S02 Top-1이 가장 높은 전략. 동률이면 현재 설정 → 목록 순서."""
    top1 = {s: sum(x.correct for x in scored(samples) if x.strategy == s) for s in strategies}
    order = sorted(strategies, key=lambda s: (-top1[s], s != current, strategies.index(s)))
    return order[0], top1


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
def _collect_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def _run_video(pipeline, bundle, path: Path, subject: str, true_label: str, strategies: list[str]) -> list[Sample]:
    outs = pipeline.run_strategies(path, strategies)
    raw = DecisionRule(0.0, 0.0)                     # 기준은 나중에 바꿔 가며 적용 → 여기선 원본 확률만
    samples = []
    for strategy, out in outs.items():
        preds = predict_tracks(
            bundle, {t.track_id: t.windows for t in out.tracks}, raw,
            modes_by_track={t.track_id: t.window_mode for t in out.tracks},
        )
        valid = {t.track_id: t.valid_frames for t in out.tracks}
        main_id = max(valid, key=valid.get) if valid else None
        if not preds and true_label:                 # 사람을 못 찾음 → 오답·불확실로 기록
            samples.append(Sample(path.name, subject, true_label, strategy, -1, True, "NO_PERSON", 0.0, 0.0, 0))
        for p in preds:
            samples.append(Sample(
                video=path.name, subject=subject, true_label=true_label, strategy=strategy,
                track_id=p.track_id, is_main=bool(true_label) and p.track_id == main_id,
                pred_label=p.action_label, confidence=p.confidence, margin=p.margin, valid_frames=valid[p.track_id],
            ))
    return samples


def _fmt(r: dict[str, int]) -> str:
    acc = f"{r['confirmed_correct']}/{r['confirmed']}" if r["confirmed"] else "확정 0건"
    return f"{r['top1']}/{r['total']} | {r['confirmed']} | {acc} | {r['uncertain']}"


def _write_report(out_dir: Path, ctx: dict) -> Path:
    lines = [
        "# Held-out 평가 — 학습하지 않은 사람(S02·S03) 영상",
        "",
        f"- 실행: {ctx['when']} · 영상 {ctx['n_videos']}개 · 처리 {ctx['elapsed']:.1f}초 · `scripts/evaluate_holdout.py`",
        f"- 모델: 6일차 RF (S01 · 6 clip 학습) · 해시 확인 {ctx['checksum']}",
        "- **S02로만** 전략·판단 기준을 고르고, **S03은 고른 설정을 한 번 적용한 결과만** 보고합니다.",
        f"- 판단 기준 점수 = 확정 정답 − {WRONG_PENALTY}×확정 오답 (오답 확정이 보류보다 나쁨) · 최고 점수 조합 {ctx['n_ties']}개 중 가운데 값",
        "",
        "## 1. Window 전략 비교 (S02 Top-1, 주인공 1명 기준)",
        "",
        "| 전략 | S02 Top-1 | 선택 |",
        "|---|---|---|",
    ]
    for s, v in ctx["strategy_top1"].items():
        lines.append(f"| `{s}` | {v}/{ctx['n_s02']} | {'✅' if s == ctx['strategy'] else ''} |")
    lines += [
        "",
        "## 2. 판단 기준 적용 결과",
        "",
        "| 설정 | 데이터 | Top-1 | 확정 | 확정 중 정답 | 판단 보류 |",
        "|---|---|---|---|---|---|",
        f"| 기존 (`{ctx['before_strategy']}`, conf ≥ {ctx['before_rule'].min_confidence}, margin ≥ {ctx['before_rule'].min_margin}) | S02 | {_fmt(ctx['before_s02'])} |",
    ]
    if ctx.get("before_s03"):
        lines.append(f"| 기존 | S03 | {_fmt(ctx['before_s03'])} |")
    lines.append(
        f"| **선택** (`{ctx['strategy']}`, conf ≥ {ctx['rule'].min_confidence}, margin ≥ {ctx['rule'].min_margin}) | S02 | {_fmt(ctx['after_s02'])} |"
    )
    if ctx.get("after_s03"):
        lines.append(f"| **선택** | **S03 (테스트)** | {_fmt(ctx['after_s03'])} |")
    if ctx.get("fallback"):
        lines += ["", "> ⚠️ S02에서 정답을 확정할 수 있는 기준 조합이 없어 **기존 기준을 유지**합니다 (임의의 값을 고르지 않음)."]
    if ctx.get("neg_after") and ctx["neg_after"]["neg_total"]:
        n = ctx["neg_after"]
        lines += ["", f"- 학습하지 않은 동작 영상: 사람 {n['neg_total']}명 중 **{n['neg_confirmed']}명을 확정**(0이어야 정상)"]

    if ctx.get("s03_rows"):
        lines += ["", "## 3. S03 영상별 결과 (선택한 설정)", "", "| 영상 | 정답 | 예측 | Confidence | 1·2등 차이 | 판정 | 결과 |", "|---|---|---|---|---|---|---|"]
        for s in ctx["s03_rows"]:
            unc = ctx["rule"].is_uncertain(s.confidence, s.margin)
            verdict = "판단 보류" if unc else "확정"
            mark = "—" if unc else ("✅" if s.correct else "❌")
            lines.append(f"| {s.video} | {s.true_label} | {s.pred_label} | {s.confidence:.3f} | {s.margin:.3f} | {verdict} | {mark} |")

    lines += [
        "",
        "## 4. 적용 방법",
        "",
        "`ai_service/project.json`에 아래 값을 넣고 fastapi를 다시 시작합니다.",
        "",
        "```json",
        json.dumps({"window": {"strategy": ctx["strategy"]}, "decision": ctx["rule"].to_dict()}, indent=2),
        "```",
        "",
        "> 표본이 사람 1명당 6개뿐이라 수치는 방향을 보는 참고값입니다. 새 촬영 데이터가 생기면 이 스크립트로 다시 고릅니다.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "holdout_eval.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S02로 고르고 S03으로 보고하는 Held-out 평가")
    ap.add_argument("--videos", type=Path, required=True, help="S02_A003_T01.mp4 형식 영상이 있는 폴더 (하위 폴더 포함)")
    ap.add_argument("--select-subject", default="S02", help="설정을 고를 사람 (기본 S02)")
    ap.add_argument("--report-subject", default="S03", help="결과만 보고할 사람 (기본 S03)")
    ap.add_argument("--negatives", type=Path, help="학습하지 않은 동작 영상 폴더 (선택)")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--out", type=Path, default=AI_ROOT.parent / "reports" / "evidence" / "holdout")
    args = ap.parse_args(argv)

    from services.model_loader import load_model_bundle
    from services.pose_pipeline import PipelineConfig, PosePipeline

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bundle = load_model_bundle()
    cfg = PipelineConfig.load()
    pipeline = PosePipeline(cfg, sequence_steps=bundle.sequence_steps, min_valid_frames=bundle.min_valid_frames)
    before_rule = DecisionRule(cfg.min_confidence, cfg.min_margin)

    jobs: list[tuple[Path, str, str]] = []
    for p in _collect_videos(args.videos.expanduser()):
        parsed = parse_video_name(p)
        if parsed and parsed[0] in (args.select_subject, args.report_subject):
            jobs.append((p, parsed[0], parsed[1]))
    if args.negatives:
        jobs += [(p, "NEG", "") for p in _collect_videos(args.negatives.expanduser())]
    if not any(s == args.select_subject for _, s, _ in jobs):
        print(f"[ERROR] {args.videos}에서 {args.select_subject}_A00x_T0x 형식 영상을 찾지 못했습니다.")
        return 1

    t0 = time.perf_counter()
    samples: list[Sample] = []
    for i, (path, subject, label) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {subject} {path.name}", flush=True)
        samples += _run_video(pipeline, bundle, path, subject, label, strategies)
    elapsed = time.perf_counter() - t0

    def pick(subject: str, strategy: str) -> list[Sample]:
        return [s for s in samples if s.subject == subject and s.strategy == strategy]

    sel = [s for s in samples if s.subject == args.select_subject]
    strategy, strategy_top1 = choose_strategy(sel, strategies, cfg.window_strategy)
    negs = pick("NEG", strategy)
    rule, chosen, n_ties = choose_rule(pick(args.select_subject, strategy) + negs)
    fallback = chosen["confirmed_correct"] == 0      # 어떤 기준으로도 정답을 확정할 수 없으면 기존 기준 유지
    if fallback:
        rule = before_rule

    ctx = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_videos": len(jobs),
        "elapsed": elapsed,
        "checksum": "✅" if bundle.checksum_verified else "— (SHA256SUMS.txt 없음)",
        "n_ties": n_ties,
        "fallback": fallback,
        "strategy_top1": strategy_top1,
        "n_s02": len({s.video for s in sel}),
        "strategy": strategy,
        "rule": rule,
        "before_strategy": cfg.window_strategy,
        "before_rule": before_rule,
        "before_s02": evaluate_rule(pick(args.select_subject, cfg.window_strategy), before_rule),
        "after_s02": evaluate_rule(pick(args.select_subject, strategy), rule),
        "neg_after": evaluate_rule(negs, rule),
    }
    report = pick(args.report_subject, strategy)
    if report:
        ctx["before_s03"] = evaluate_rule(pick(args.report_subject, cfg.window_strategy), before_rule)
        ctx["after_s03"] = evaluate_rule(report, rule)
        ctx["s03_rows"] = sorted((s for s in report if s.is_main), key=lambda s: s.video)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "subject", "true_label", "strategy", "track_id", "is_main", "pred_label",
                    "confidence", "margin", "valid_frames", "correct"])
        for s in samples:
            w.writerow([s.video, s.subject, s.true_label, s.strategy, s.track_id, s.is_main, s.pred_label,
                        s.confidence, s.margin, s.valid_frames, s.correct])
    md = _write_report(args.out, ctx)

    print()
    print(md.read_text(encoding="utf-8"))
    print(f"[저장] {md}\n[저장] {args.out / 'predictions.csv'}")
    print(f"[참고] 최고 점수 조합 {n_ties}개 중앙값 선택 · 표본 중앙값 confidence="
          f"{statistics.median([s.confidence for s in scored(sel)]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
