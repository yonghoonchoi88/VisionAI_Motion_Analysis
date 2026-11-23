# 다이어그램 원본

README에 들어가는 그림 3장의 원본입니다. 수정한 뒤 PNG로 다시 렌더링해 `docs/images/`에 덮어씁니다.

| 그림 | 원본 | 결과 |
|---|---|---|
| 전체 아키텍처 | `architecture.html` (배치 직접 지정) | `docs/images/architecture.png` |
| ERD | `erd.mmd` (Mermaid) | `docs/images/erd.png` |
| AI 분석 버튼 흐름 | `button_flow.html` (배치 직접 지정) | `docs/images/button_flow.png` |

렌더링 방법
- HTML 원본: 브라우저로 열고 `#canvas` 영역을 2배율로 캡처 (Chrome 개발자 도구 → 요소 선택 → *Capture node screenshot*)
- `erd.mmd`: [Mermaid Live Editor](https://mermaid.live)에 붙여 넣고 PNG로 내보내기

README에서 Mermaid 코드 대신 이미지를 쓰는 이유: GitHub이 Mermaid 위에 확대·이동 버튼을 겹쳐 그려 그림을 가리고, 자동 배치로는 화살표 라벨이 겹쳤기 때문입니다.
