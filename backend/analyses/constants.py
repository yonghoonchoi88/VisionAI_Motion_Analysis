"""과제 문서의 Action 표와 상태 표기를 한 곳에서 관리한다."""

# (label, action_id, 한국어 이름) — 과제 문서 순서(A001~A006)
ACTIONS = [
    ("bend_return", "A001", "허리를 굽혔다 펴기"),
    ("leg_raise_lower", "A002", "한쪽 다리를 들었다 내리기"),
    ("walk_turn_walk", "A003", "걷다가 방향을 바꿔 다시 걷기"),
    ("drink_return", "A004", "음료수를 마신 뒤 제자리로 돌아오기"),
    ("sit_stand", "A005", "앉았다 일어나기"),
    ("wave", "A006", "손 흔들기"),
]
ACTION_ID = {label: aid for label, aid, _ in ACTIONS}
ACTION_KO = {label: ko for label, _, ko in ACTIONS}
ACTION_CHOICES = [(label, f"{aid} {ko}") for label, aid, ko in ACTIONS]
