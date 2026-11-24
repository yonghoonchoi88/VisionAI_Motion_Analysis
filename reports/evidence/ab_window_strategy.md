# Window 전략 A/B 비교 (2026-11-23)

**문제**: walk_turn_walk(10.6초) 영상이 drink_return(0.20)으로 오분류
**원인**: 6일차 학습은 Clip 전체를 30 Step으로 리샘플, 추론은 7일차 방식대로 5초 Window 24개로 잘라 평균 → 동작의 일부만 담긴 입력
**변경**: project.json `window.strategy` 추가 (sliding / whole_track / auto)
**검증**: 같은 영상 7개를 두 방식으로 추론 (원본 day04 6개 + SBU 2인 영상 1개)

| 영상 | A: sliding | B: whole_track |
|---|---|---|
| bend_return | ✅ 0.65 | ✅ 0.65 |
| drink_return | ✅ 0.48 | ✅ 0.63 |
| leg_raise_lower (ID 분리) | ✅ 0.32 / 0.30 | ✅ 0.32 / 0.30 |
| sit_stand | ✅ 0.65 | ✅ 0.67 |
| walk_turn_walk | ❌ drink_return 0.20 | ✅ 0.66 |
| wave | ✅ 0.45 | ✅ 0.64 |
| 01_sbu_shaking_hands (미학습 동작, 2인) | 0.23 / 0.25 | 0.23 / 0.25 |

**결정**
- 기본 전략 `auto`: 사람이 보인 시간 ≤ 12초 → whole_track, 초과 → sliding
- 불확실 기준 0.55 유지: 전체 동작 정답(0.63~0.67)과 ID 분리 조각·미학습 동작(0.23~0.32)을 이 표본에서 정확히 구분

**주의**: day04 영상은 학습 원본일 가능성이 높아 정확도 증거가 아니라 '이관 파이프라인 회귀 확인'임
**알려진 한계**: leg_raise_lower에서 ByteTrack ID가 바뀌어 한 사람이 track 1·3으로 분리됨

---

**후속 (2026-11-24)** — 위 비교는 학습 원본일 가능성이 높은 day04 영상이라, 학습하지 않은 사람(S02)으로 4가지 전략을 다시 비교했습니다: `auto`·`whole_track` 3/6, `sliding`·`hybrid` 4/6 → 기본 전략을 **`sliding`**으로 변경. S03 결과와 판단 기준은 [`holdout/holdout_eval.md`](holdout/holdout_eval.md).
