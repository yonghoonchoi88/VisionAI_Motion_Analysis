# 문제 해결 기록

구현하면서 실제로 만난 문제 7건을 **문제 → 확인 → 변경 → 검증** 순서로 기록했습니다. 과제 발표 6번("어떤 오류를 만났고 어떻게 해결했나요?")의 근거 자료입니다.

| # | 분류 | 한 줄 요약 |
|---|---|---|
| [①](#t1) | 모델 이관 | 학습은 Clip 전체, 추론은 5초 Window → 10초 영상 오분류 |
| [②](#t2) | Docker | CPU torch + PyPI torchvision → YOLO 로드 실패 |
| [③](#t3) | Docker | `packaging` 누락 → Celery Worker 재시작 반복 |
| [④](#t4) | Docker | 포트 충돌 + 중단된 첫 실행의 컨테이너 |
| [⑤](#t5) | 화면 | OpenCV `mp4v`는 Chrome에서 재생 불가 |
| [⑥](#t6) | 성능 | 스트림 버퍼 몰림 + VP8 인코딩 → 44초 |
| [⑦](#t7) | 데이터 정확성 | 벽시계로 경과 시간을 재서 시각 조정에 취약 |

---

<a id="t1"></a>

## ① 학습과 추론의 입력 단위 불일치 — 10초 영상 오분류

| 단계 | 내용 |
|---|---|
| **문제** | 원본 day04 영상 6개를 추론했더니 `walk_turn_walk`(10.6초)만 `drink_return` 0.20으로 틀림 |
| **확인** | 6일차 학습(`34_build_sequence_dataset.py`)은 **동작 Clip 전체**를 30 Step으로 리샘플. 그런데 추론은 7일차 실시간 방식대로 **최근 5초 Window 24개**를 만들어 확률을 평균 → Window마다 "걷기만", "돌기만"처럼 동작의 일부만 담겨 학습 때 본 적 없는 입력이 됨 |
| **변경** | `ai_service/project.json`에 `window.strategy` 추가: `sliding` / `whole_track` / `auto`. 기본 `auto` = 사람이 보인 시간이 12초 이하면 전체 구간 1개 Window |
| **검증** | 같은 7개 영상으로 A/B 비교: sliding 5/6 → **whole_track 6/6**, walk_turn_walk 0.20 오답 → **0.66 정답**, drink_return 0.48 → 0.63, wave 0.45 → 0.64 |
| **교훈** | 모델 파일만 옮긴다고 끝이 아니다. **학습 때의 전처리 단위까지 맞춰야** 같은 성능이 나온다 (training-serving skew) |

근거: [`reports/evidence/ab_window_strategy.md`](../reports/evidence/ab_window_strategy.md)

<a id="t2"></a>

## ② torch와 torchvision 빌드 불일치 — fastapi가 healthy가 되지 않음

| 단계 | 내용 |
|---|---|
| **문제** | `docker compose up` 후 fastapi가 계속 `health: starting`, 그래서 django·celery도 시작하지 않음 |
| **확인** | `/health` 응답의 `error`와 로그에 `RuntimeError: operator torchvision::nms does not exist`. Dockerfile이 torch만 CPU 전용 인덱스에서 받고, ultralytics가 필요로 하는 torchvision은 일반 PyPI에서 따로 설치 → 서로 다른 빌드가 섞임 |
| **변경** | `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision` — **같은 인덱스에서 함께** 설치 |
| **검증** | 컨테이너 안에서 두 버전 모두 `+cpu` 확인, `/health` → `model_ready: true`, fastapi healthy |
| **교훈** | 모델 로드에 실패해도 서버는 띄우고 원인을 `/health`로 보고하게 만든 설계 덕분에 원인을 바로 찾을 수 있었다 |

<a id="t3"></a>

## ③ 개발 PC에 우연히 있던 패키지에 의존 — Celery 재시작 반복

| 단계 | 내용 |
|---|---|
| **문제** | celery 컨테이너가 `Restarting (1)` 반복 |
| **확인** | `ModuleNotFoundError: No module named 'packaging'` (kombu의 Redis 전송 모듈이 필요). 개발 환경에는 이 패키지가 우연히 깔려 있어 테스트가 통과했음. **깨끗한 가상환경에서 `requirements.txt`만으로 설치해 같은 오류를 재현** |
| **변경** | `celery` → `celery[redis]` (Redis 전송에 필요한 의존성을 함께 설치) |
| **검증** | 깨끗한 가상환경에서 Django check·Celery 로드·테스트 통과, 컨테이너에서 `celery@… ready.` |
| **교훈** | "내 PC에서는 된다"는 증거가 아니다. **깨끗한 환경에서 재현**해야 하고, Docker 이미지가 그 검증 역할을 한다 |

<a id="t4"></a>

## ④ 포트 충돌과 중단된 컨테이너 — django가 뜨지 않음

| 단계 | 내용 |
|---|---|
| **문제** | `Bind for 0.0.0.0:8001 failed: port is already allocated` (이후 8000에서도 같은 오류), 이어서 django 로그에 `failed to resolve host 'postgres'` |
| **확인** | 다른 프로젝트가 같은 포트를 사용 중. 첫 실행이 중단되면서 만들어진 django·celery 컨테이너가 네트워크에 제대로 연결되지 않은 상태로 재사용됨 |
| **변경** | `compose.yaml`의 바깥 포트를 `${DJANGO_HOST_PORT:-8000}` 등 **환경변수**로 바꾸고 `.env`에서 8010·8011로 지정, `--force-recreate`로 컨테이너 재생성 |
| **검증** | 5개 서비스 모두 Up, 컨테이너끼리는 서비스 이름(`postgres`, `fastapi`)으로 정상 통신 |
| **교훈** | 바깥 포트는 환경마다 다르다 — 설정으로 빼 두면 `.env` 한 줄로 해결된다. 컨테이너 사이 통신은 포트 변경과 무관하다 |

<a id="t5"></a>

## ⑤ 수집 영상이 브라우저에서 재생되지 않음

| 단계 | 내용 |
|---|---|
| **문제** | IP Webcam으로 수집한 영상이 상세 화면에서 `0:00`으로 멈춤. 업로드한 day04 영상은 정상 재생 |
| **확인** | OpenCV 기본 코덱 `mp4v`(MPEG-4 Part 2)는 Chrome이 지원하지 않음(MediaError 4). 업로드 원본은 H.264라 재생됐던 것. Chrome으로 직접 비교: mp4v 실패 / **WebM(VP8) 재생 성공** |
| **변경** | 수집 영상을 **WebM(VP8)** 으로 저장, VP8을 쓸 수 없는 환경이면 MP4로 대체 |
| **검증** | 상세 화면에서 재생·길이 정상, FastAPI도 WebM을 그대로 읽음, 테스트에 `.webm` 확인 추가 |

<a id="t6"></a>

## ⑥ IP Webcam 전체 처리 44초

| 단계 | 내용 |
|---|---|
| **문제** | 8초 수집인데 전체 처리 44초, 영상 FPS가 55로 기록됨 |
| **확인** | 휴대폰 스트림이 쌓인 Frame을 몰아서 보내 Frame이 과도하게 많았고, VP8 인코딩 시간은 Frame 수 × 해상도에 비례 (측정: 440장·960px 83초 vs 240장·640px 24초) |
| **변경** | **30fps 간격**으로만 저장, **가로 640px**로 저장 (YOLO가 어차피 640으로 줄이므로 정확도 손해 없음) |
| **검증** | 전체 44.0초 → **17.7초**, AI 추론 11.9초 → 4.3초. 실제 카메라처럼 실시간 속도로 Frame을 내보내는 가짜 스트림으로 테스트 재작성 |

<a id="t7"></a>

## ⑦ 전체 처리 시간이 AI 추론 시간보다 짧게 표시됨

| 단계 | 내용 |
|---|---|
| **문제** | 비교 화면에서 #3 bend_return이 **전체 0.1초 / AI 2.4초** — 전체는 AI 시간을 포함하므로 불가능한 값 |
| **확인** | DB 시각 확인: 다른 작업은 `started → finished` 간격 ≈ AI 시간인데, #3만 0.13초. AI 추론(FastAPI의 단조 시계 측정)은 다른 영상과 비율이 맞음 → **Celery 컨테이너의 벽시계가 처리 도중 약 2.2초 뒤로 조정**된 것으로 판단 (WSL2·Docker는 호스트 시계와 동기화하며 시각을 조정할 수 있음) |
| **변경** | 경과 시간을 벽시계 두 시각의 차가 아니라 **단조 시계(`time.monotonic()`)** 로 측정해 `processing_time_sec`에 저장(마이그레이션 0002). 전체 = 대기 + Worker 처리 시간. 이전 기록은 AI 시간을 하한으로 사용 |
| **검증** | 벽시계를 일부러 0.04초 차이로 조작하고 실제 처리는 0.3초 걸리게 한 테스트에서 전체 시간 ≥ 0.3초. 화면의 #3이 2.4초로 표시됨 |
| **교훈** | **시각(언제)** 은 벽시계로, **경과 시간(얼마나)** 은 단조 시계로 잰다. 화면의 숫자를 그대로 믿지 않고 의심한 것이 발견의 시작이었다 |

---

## 참고: 개선되지 않은 실험도 결과다

IP Webcam 신뢰도가 0.3 안팎인 원인으로 "수집 구간에 정지 동작이 섞였다"를 의심해 조건을 하나씩 바꿨지만(8초 쉼 / 8초 연속 / 5초 연속) 3회 모두 wave 1순위에 신뢰도 0.26~0.30으로 개선이 없어 **가설을 기각**했습니다. 원인은 학습 데이터(3명·특정 장소)와 새 사람·새 환경의 차이이며, 재학습 없이는 개선할 수 없습니다.

근거: [`reports/evidence/ab_capture_condition.md`](../reports/evidence/ab_capture_condition.md)
