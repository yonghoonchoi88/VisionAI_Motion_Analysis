"""README '한눈에 보기' + 발표 첫 화면용 이미지(16:9, 1600×900) 생성기 — 한국어 / English.

실행 (레포 루트):
    python docs/diagrams/make_overview.py            # docs/images/overview.svg · overview.en.svg
    for f in overview overview.en; do rsvg-convert -w 1600 docs/images/$f.svg -o docs/images/$f.png; done

숫자는 README · reports/evidence 에 있는 검증된 값만 쓴다. 값이 바뀌면 TEXT만 고치고 다시 실행한다.
"""
from pathlib import Path

OUT = Path("docs/images")
W, H = 1600, 900
FONT = "'Noto Sans CJK KR','Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif"

# 색: 본문 네이비 + 강조 파랑 1개 + 회색(이전 방식). 색은 의미가 있을 때만 쓴다.
INK, MUTED, LINE = "#0F172A", "#64748B", "#E2E8F0"
ACCENT, ACCENT_BG = "#1D4ED8", "#EFF6FF"
PANEL, OLD_BG = "#F8FAFC", "#F1F5F9"

TEXT = {
    "ko": {
        "subtitle": "6일차 동작 인식 모델을 재학습 없이 웹 서비스로 · 영상 속 사람마다 6개 동작 판단",
        "course": "교과10 개인 과제",
        "kpis": [
            ("기능 테스트 T01–T10", "10 / 10", "모두 캡처·로그로 증거 보관"),
            ("자동 테스트", "80개", "FastAPI 45 · Django 35"),
            ("분석 요청 응답", "0.1초 이내", "202 즉시 응답 → 비동기 처리"),
            ("학습 안 한 사람 S03", "확정 2 / 2", "확정한 판단은 모두 정답"),
        ],
        "flow_title": "버튼 한 번의 흐름",
        "flow_sub": "Docker Compose 5개 서비스",
        "nodes": {
            "input": ("입력", ["영상 업로드", "IP Webcam 8초"]),
            "django": ("Django + DRF", ["작업 저장", "커밋 후 등록"]),
            "redis": ("Redis", ["작업 대기열"]),
            "worker": ("Celery Worker", ["실패 시 재시도", "3회 × 10초"]),
            "db": ("PostgreSQL", ["작업 1건 : 사람별 예측 N건"]),
            "fastapi": ("FastAPI 추론", ["YOLO11n-pose → RF", "DB 접근 없음"]),
        },
        "save": "결과 저장",
        "flow_caption": "브라우저는 1초마다 상태 조회 · Worker를 25초 멈췄다 켜도 작업 유실 없음",
        "prob_title": "가장 큰 문제: '판단 불확실' 기준",
        "old": ("이전 · 신뢰도 ≥ 0.55 (7일차 방식)", "S03 확정 0건", "1등은 6개 모두 맞혔는데 전부 보류"),
        "new": ("개선 · 신뢰도 ≥ 0.20 + 1·2등 차이 ≥ 0.11", "확정 2건 · 모두 정답",
                "학습 안 한 동작 0 / 6 확정", "기준은 검증용 S02로만 고르고, S03은 보고만"),
        "chips_title": "믿을 수 있게 만든 장치",
        "chips": ["SHA256 모델 검증", "Feature 순서 고정 (뒤집으면 0.50→0.17)", "커밋 후 작업 등록",
                  "acks_late + 재시도", "다른 사용자 작업 404", "카메라 주소 허용 목록"],
    },
    "en": {
        "subtitle": "Day-6 action model served on the web without retraining · 6 actions per person in a video",
        "course": "Course 10 · individual project",
        "kpis": [
            ("Functional tests T01–T10", "10 / 10", "Each backed by screenshots · logs"),
            ("Automated tests", "80", "FastAPI 45 · Django 35"),
            ("Job submit response", "≤ 0.1 s", "202 at once → async processing"),
            ("Unseen person S03 · confirmed", "2 / 2", "Both confirmed calls correct"),
        ],
        "flow_title": "One button, end to end",
        "flow_sub": "5 Docker Compose services",
        "nodes": {
            "input": ("Input", ["Video upload", "IP Webcam 8 s"]),
            "django": ("Django + DRF", ["Save job", "Enqueue on commit"]),
            "redis": ("Redis", ["Job queue"]),
            "worker": ("Celery Worker", ["Retry on failure", "3 × 10 s"]),
            "db": ("PostgreSQL", ["1 job : N per-person rows"]),
            "fastapi": ("FastAPI inference", ["YOLO11n-pose → RF", "No DB access"]),
        },
        "save": "save results",
        "flow_caption": "Browser polls every 1 s · Worker stopped for 25 s and restarted — no job lost",
        "prob_title": "Key problem: the 'uncertain' rule",
        "old": ("Before · confidence ≥ 0.55 (Day-7 rule)", "S03: 0 confirmed", "Top-1 right on all 6, yet all held back"),
        "new": ("After · confidence ≥ 0.20 + top-1/2 gap ≥ 0.11", "2 confirmed · both right",
                "Unseen actions: 0 / 6 confirmed", "Thresholds picked on S02 only; S03 only reported"),
        "chips_title": "What makes it trustworthy",
        "chips": ["SHA256 model check", "Fixed feature order (reversed: 0.50→0.17)", "Enqueue after commit",
                  "acks_late + retry", "Others' jobs → 404", "Camera host allowlist"],
    },
}


def text_w(s: str, size: float) -> float:
    """대략적인 글자 폭 추정 (한글 1em, 영문/숫자 0.58em) — 칩 폭 계산용."""
    return sum(size if ord(c) > 0x2E80 else size * 0.58 for c in s)


def build(lang: str) -> str:
    T = TEXT[lang]
    out: list[str] = []

    def t(x, y, s, size=20, color=INK, weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                   f'text-anchor="{anchor}">{s}</text>')

    def rect(x, y, w, h, fill, stroke=LINE, r=14):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')

    def arrow(points, color=MUTED):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" marker-end="url(#ah)"/>')

    def node(x, y, w, h, key, accent=False):
        title, lines = T["nodes"][key]
        rect(x, y, w, h, ACCENT_BG if accent else "#FFFFFF", ACCENT if accent else "#CBD5E1", 12)
        cx = x + w / 2
        t(cx, y + 34, title, 20, ACCENT if accent else INK, 700, "middle")
        for i, ln in enumerate(lines):
            t(cx, y + 60 + i * 22, ln, 16, MUTED, 400, "middle")

    # ── 헤더 ──
    t(64, 80, "VisionAI Motion Analysis", 46, INK, 800)
    t(64, 120, T["subtitle"], 22, MUTED)
    t(1536, 80, T["course"], 20, MUTED, 500, "end")
    t(1536, 112, "Django/DRF · Celery · Redis · FastAPI · PostgreSQL", 18, MUTED, 400, "end")

    # ── KPI 4칸 ──
    kw, gap, ky = 350, 24, 156
    for i, (label, big, sub) in enumerate(T["kpis"]):
        x = 64 + i * (kw + gap)
        rect(x, ky, kw, 168, "#FFFFFF")
        out.append(f'<rect x="{x}" y="{ky}" width="6" height="168" rx="3" fill="{ACCENT}"/>')
        t(x + 30, ky + 42, label, 20, MUTED, 500)
        t(x + 30, ky + 108, big, 50, ACCENT, 800)
        t(x + 30, ky + 146, sub, 18, MUTED)

    # ── 왼쪽: 버튼 한 번의 흐름 ──
    rect(64, 352, 900, 392, PANEL)
    t(96, 396, T["flow_title"], 24, INK, 700)
    t(96 + text_w(T["flow_title"], 24) + 30, 396, T["flow_sub"], 18, MUTED)
    ay, by, bh = 424, 588, 100
    node(96, ay, 180, bh, "input")
    node(316, ay, 190, bh, "django")
    node(546, ay, 140, bh, "redis")
    node(726, ay, 206, bh, "worker")
    for x1, x2 in [(276, 316), (506, 546), (686, 726)]:
        arrow([(x1 + 2, ay + bh / 2), (x2 - 6, ay + bh / 2)])
    node(316, by, 210, bh, "db")
    node(726, by, 206, bh, "fastapi", accent=True)
    arrow([(880, ay + bh), (880, by - 6)])                          # Worker → FastAPI
    arrow([(770, ay + bh), (770, 556), (440, 556), (440, by - 6)])  # Worker → DB 저장
    arrow([(360, ay + bh), (360, by - 6)])                          # Django ↔ DB
    t(600, 548, T["save"], 15, MUTED, 400, "middle")
    t(96, 724, T["flow_caption"], 18, MUTED)

    # ── 오른쪽: 가장 큰 문제 → 해결 ──
    rx, rw = 988, 548
    rect(rx, 352, rw, 392, PANEL)
    t(rx + 32, 396, T["prob_title"], 24, INK, 700)
    o1, o2, o3 = T["old"]
    rect(rx + 32, 420, rw - 64, 124, OLD_BG, "#CBD5E1", 12)
    t(rx + 56, 454, o1, 19, MUTED, 500)
    t(rx + 56, 500, o2, 36, MUTED, 800)
    t(rx + 56, 530, o3, 17, MUTED)
    arrow([(rx + rw / 2, 548), (rx + rw / 2, 574)], ACCENT)
    n1, n2, n3, n4 = T["new"]
    rect(rx + 32, 580, rw - 64, 146, ACCENT_BG, ACCENT, 12)
    t(rx + 56, 614, n1, 19, ACCENT, 600)
    t(rx + 56, 662, n2, 36, ACCENT, 800)
    t(rx + 56, 688, n3, 17, INK)
    t(rx + 56, 712, n4, 15, MUTED)

    # ── 하단: 신뢰성 장치 칩 (한 줄에 안 들어가면 글자 크기를 줄인다) ──
    t(64, 800, T["chips_title"], 20, INK, 700)
    size = 17
    while sum(text_w(c, size) + 48 for c in T["chips"]) - 12 > 1472:
        size -= 0.5
    cx = 64
    for c in T["chips"]:
        w = text_w(c, size) + 36
        rect(cx, 820, w, 44, "#FFFFFF", "#CBD5E1", 22)
        t(cx + w / 2, 849, c, size, INK, 500, "middle")
        cx += w + 12

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">
<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>
{chr(10).join(out)}
</svg>"""


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, name in [("ko", "overview.svg"), ("en", "overview.en.svg")]:
        (OUT / name).write_text(build(lang), encoding="utf-8")
        print("wrote", OUT / name)
