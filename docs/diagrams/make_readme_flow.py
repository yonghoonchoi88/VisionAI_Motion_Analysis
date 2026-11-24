"""README '한눈에 보기' 흐름 이미지 — Soybean / kimchi 레포와 같은 스타일(색·폰트·카드 구조).

출력: docs/images/readme_flow.svg (원본) · readme_flow.png (README용)
"""
import sys
from pathlib import Path

# 실행: python docs/diagrams/make_readme_flow.py docs/images && rsvg-convert docs/images/readme_flow.svg -o docs/images/readme_flow.png
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/images")
FONT = "'Noto Sans CJK KR','Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif"
INK, SUB, BORDER, ARROW, BG = "#1f2328", "#57606a", "#d0d7de", "#8c959f", "#f6f5f2"
GRAY, BLUE, ORANGE, GREEN = "#6e7781", "#2f6fdb", "#e8663d", "#1a7f37"

# (제목, 설명, 핵심 숫자, 색, 카드 폭) — kimchi 레포와 같은 3줄 카드
cards = [
    ("입력 영상", "업로드 · IP Webcam 8초", "사람마다 Track 분리", GRAY, 300),
    ("① AI 추론 · FastAPI", "YOLO11n-pose → 30×38 → RF", "재학습 0회 · SHA256 확인", BLUE, 400),
    ("② 웹 서비스 · Django", "202 접수 → Celery → DB", "T01–T10 10/10 · 테스트 80", ORANGE, 420),
    ("검증 · 새 사람 S03", "S02로 기준 선택 → S03 보고", "확정 2/2 정답", GREEN, 330),
]
W, H, X0, Y, CH, GAP = 1760, 410, 50, 50, 220, 70


def tw(s, size):  # 대략적 글자 폭 (한글 1em · 그 외 0.58em)
    return sum(size if ord(c) > 0x2E80 else size * 0.58 for c in s)


o = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">',
    f'<rect width="{W}" height="{H}" rx="24" fill="{BG}"/>',
    '<defs>'
    f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>'
    f'<marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>'
    '</defs>',
]
x, centers = X0, []
for title, desc, num, color, w in cards:
    cx = x + w / 2
    centers.append((x, cx, w))
    for s, size in [(title, 32), (desc, 22), (num, 24)]:
        if tw(s, size) > w - 24: print(f"확인 필요: {s} ({tw(s, size):.0f} > {w - 24})")
    o += [
        f'<rect x="{x}" y="{Y}" width="{w}" height="{CH}" rx="18" fill="#ffffff" stroke="{BORDER}" stroke-width="2"/>',
        f'<rect x="{x}" y="{Y}" width="{w}" height="10" rx="5" fill="{color}"/>',
        f'<text x="{cx}" y="132" text-anchor="middle" font-size="32" font-weight="700" fill="{INK}">{title}</text>',
        f'<text x="{cx}" y="186" text-anchor="middle" font-size="22" fill="{SUB}">{desc}</text>',
        f'<text x="{cx}" y="236" text-anchor="middle" font-size="24" font-weight="700" fill="{color}">{num}</text>',
    ]
    x += w + GAP
assert x - GAP + X0 <= W, f"캔버스 폭 부족: {x - GAP + X0}"

for (x1, _, w1), (x2, _, _) in zip(centers, centers[1:]):  # 카드 사이 화살표
    o.append(f'<line x1="{x1 + w1 + 8}" y1="160" x2="{x2 - 8}" y2="160" stroke="{ARROW}" stroke-width="4" marker-end="url(#a)"/>')

# 검증 → ① 로 돌아가는 점선: 판단 기준을 고친 이야기
last_cx, first_model_cx = centers[3][1], centers[1][1]
label = "판단 기준 개선 · 신뢰도 0.55 → 1·2등 차이도 함께"
lw = tw(label, 25) + 60
mid = (last_cx + first_model_cx) / 2
o += [
    f'<path d="M{last_cx},278 V360 H{first_model_cx} V282" fill="none" stroke="{GREEN}" stroke-width="4" stroke-dasharray="12 8" marker-end="url(#g)"/>',
    f'<rect x="{mid - lw / 2}" y="334" width="{lw}" height="52" rx="26" fill="{BG}"/>',
    f'<text x="{mid}" y="369" text-anchor="middle" font-size="25" font-weight="700" fill="{GREEN}">{label}</text>',
    "</svg>",
]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "readme_flow.svg").write_text("\n".join(o), encoding="utf-8")
print("ok")
