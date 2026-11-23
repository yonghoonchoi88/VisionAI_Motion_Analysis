"""
6일차 학습 산출물 로더 (model_loader.py)

역할
    세 파일(학습 모델 · Feature 순서 · 학습 정보)을 읽고, 세 파일의 규격이 서로 맞는지
    교차 검증한 뒤 ModelBundle 하나로 묶어 반환한다.

사용 방식
    - FastAPI 시작 시 load_model_bundle()을 1회 호출하고, 이후 모든 요청에서 재사용한다.
    - 이 파일을 직접 실행하면(T01) 검증 결과를 출력한다.
          python -m services.model_loader          (ai_service/ 폴더에서 실행)

보안 메모
    joblib.load()는 내부적으로 pickle을 사용하므로 임의 코드가 실행될 수 있다.
    반드시 본인이 학습한 신뢰 가능한 파일에만 사용한다. (사용자 업로드 파일에 절대 사용 금지)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import sklearn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 경로·규격 상수
# ─────────────────────────────────────────────────────────────
# 실행 위치(cwd)와 무관하게 ai_service/models/ 를 가리키는 절대경로.
# uvicorn을 어디서 실행하든, Docker 컨테이너 안이든 깨지지 않는다.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILE = "rf_action_baseline.joblib"
FEATURE_COLUMNS_FILE = "rf_feature_columns.json"
TRAINING_INFO_FILE = "rf_training_info.json"

FEATURES_PER_STEP = 38      # COCO 17 Keypoint × (x, y) = 34 + 관절 각도 4개(팔꿈치·무릎 좌우)
EXPECTED_NUM_CLASSES = 6

# 과제 문서의 Action ID 표.
# model.classes_ 는 알파벳순이라 과제 문서 순서(A001~A006)와 다르다.
# → 위치(index)가 아니라 반드시 '이름'으로 연결한다.
ACTION_IDS: dict[str, str] = {
    "bend_return": "A001",
    "leg_raise_lower": "A002",
    "walk_turn_walk": "A003",
    "drink_return": "A004",
    "sit_stand": "A005",
    "wave": "A006",
}

REQUIRED_INFO_KEYS = ("classes", "feature_count", "sequence_steps", "min_valid_frames")


# ─────────────────────────────────────────────────────────────
# 예외·데이터 구조
# ─────────────────────────────────────────────────────────────
class ModelSpecError(RuntimeError):
    """모델 산출물이 없거나 규격이 맞지 않음 — 이 상태로는 추론 서버를 띄우면 안 된다."""


@dataclass(frozen=True)  # frozen: 서버 실행 중 실수로 값이 바뀌지 않게 (C++의 const 객체와 비슷)
class ModelBundle:
    model: Any                          # sklearn RandomForestClassifier
    feature_columns: tuple[str, ...]    # 1140개, 학습 때 순서 그대로
    classes: tuple[str, ...]            # model.classes_ 순서 = predict_proba() 열 순서
    sequence_steps: int                 # 30
    features_per_step: int              # 38
    min_valid_frames: int               # 10 — 이보다 짧게 검출된 사람은 판단하지 않음
    expects_dataframe: bool             # 학습 시 DataFrame을 썼다면 True (feature_names_in_ 존재)
    training_info: dict[str, Any] = field(repr=False, hash=False, compare=False)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def action_id(self, label: str) -> str:
        """Action 이름 → 과제 ID (예: 'wave' → 'A006')."""
        return ACTION_IDS[label]

    def summary(self) -> dict[str, Any]:
        """/health 응답·로그에 쓰는 요약 정보 (모델 객체 자체는 제외)."""
        return {
            "model_type": type(self.model).__name__,
            "sklearn_version": sklearn.__version__,
            "n_features": self.n_features,
            "sequence_steps": self.sequence_steps,
            "features_per_step": self.features_per_step,
            "min_valid_frames": self.min_valid_frames,
            "classes": [{"label": c, "action_id": ACTION_IDS[c]} for c in self.classes],
        }


# ─────────────────────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────────────────────
def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ModelSpecError(f"JSON 형식 오류: {path.name} ({e})") from e


def _load_model(path: Path) -> Any:
    """joblib 로드. scikit-learn 버전 불일치 경고는 '오류'로 격상한다.

    버전이 다른 모델은 로드는 되더라도 예측값이 달라질 수 있어 서비스에 쓰면 안 된다.
    """
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(path)

    for w in caught:
        if issubclass(w.category, InconsistentVersionWarning):
            raise ModelSpecError(
                f"scikit-learn 버전 불일치: 학습 {w.message.original_sklearn_version} "
                f"/ 현재 {w.message.current_sklearn_version} → requirements.txt 버전을 맞추세요."
            )
    return model


def _validate(model: Any, feature_columns: Any, info: Any) -> list[str]:
    """세 파일을 교차 검증하고, 발견한 문제를 '전부' 모아 반환한다 (없으면 빈 리스트).

    첫 오류에서 멈추지 않고 모두 모으는 이유: 한 번 실행으로 고칠 목록을 전부 보기 위함.
    """
    errors: list[str] = []

    # ── 1. 각 파일의 기본 형태 ──────────────────────────────
    for attr in ("predict_proba", "classes_", "n_features_in_"):
        if not hasattr(model, attr):
            errors.append(f"모델에 '{attr}' 속성이 없습니다 (학습 완료된 분류기가 아님).")

    if not isinstance(feature_columns, list) or not all(isinstance(c, str) for c in feature_columns):
        errors.append(f"{FEATURE_COLUMNS_FILE}는 문자열 리스트여야 합니다.")
        return errors  # 이후 검사가 불가능하므로 여기서 반환

    if not isinstance(info, dict):
        errors.append(f"{TRAINING_INFO_FILE}는 JSON 객체(dict)여야 합니다.")
        return errors

    missing_keys = [k for k in REQUIRED_INFO_KEYS if k not in info]
    if missing_keys:
        errors.append(f"{TRAINING_INFO_FILE}에 필수 키가 없습니다: {missing_keys}")
        return errors

    if errors:
        return errors

    n_cols = len(feature_columns)
    steps = info["sequence_steps"]

    # ── 2. Feature 개수: 모델 = 컬럼 파일 = 학습 정보 = Step × 38 ──
    if not (model.n_features_in_ == n_cols == info["feature_count"]):
        errors.append(
            f"Feature 수 불일치: model.n_features_in_={model.n_features_in_}, "
            f"feature_columns={n_cols}, info.feature_count={info['feature_count']}"
        )
    if n_cols != steps * FEATURES_PER_STEP:
        errors.append(f"Feature 수({n_cols}) ≠ sequence_steps({steps}) × {FEATURES_PER_STEP}")

    if len(set(feature_columns)) != n_cols:
        errors.append("feature_columns에 중복 이름이 있습니다.")

    # ── 3. Feature 배치: i번째 컬럼은 반드시 t{i // 38:02d}_ 로 시작 (Step-major 순서) ──
    bad_layout = [
        (i, c) for i, c in enumerate(feature_columns)
        if not c.startswith(f"t{i // FEATURES_PER_STEP:02d}_")
    ]
    if bad_layout:
        errors.append(f"Feature 배치가 Step 순서가 아닙니다. 첫 위반: index {bad_layout[0][0]} = '{bad_layout[0][1]}'")

    # ── 4. Step마다 같은 38개 이름이 같은 순서로 반복되는지 ──
    if not bad_layout and n_cols == steps * FEATURES_PER_STEP:
        step0 = [c[4:] for c in feature_columns[:FEATURES_PER_STEP]]   # 't00_' 제거
        for s in range(1, steps):
            chunk = feature_columns[s * FEATURES_PER_STEP:(s + 1) * FEATURES_PER_STEP]
            if [c[4:] for c in chunk] != step0:
                errors.append(f"Step {s:02d}의 Feature 구성·순서가 Step 00과 다릅니다.")
                break

    # ── 5. 모델이 이름 정보를 갖고 있다면 순서까지 완전히 같아야 함 ──
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != feature_columns:
        errors.append("model.feature_names_in_ 순서가 feature_columns.json과 다릅니다.")

    # ── 6. Class: 개수 6개, 학습 정보와 구성 일치, 과제 ID 표와 일치 ──
    model_classes = [str(c) for c in model.classes_]
    if len(model_classes) != EXPECTED_NUM_CLASSES:
        errors.append(f"Class 수가 {EXPECTED_NUM_CLASSES}개가 아닙니다: {model_classes}")
    if set(model_classes) != set(info["classes"]):   # 순서가 아니라 '구성'을 비교
        errors.append(f"Class 구성 불일치: model={model_classes}, info={info['classes']}")
    unknown = set(model_classes) - set(ACTION_IDS)
    if unknown:
        errors.append(f"과제 Action ID 표에 없는 Class: {sorted(unknown)}")

    # ── 7. min_valid_frames 값 범위 ──
    mvf = info["min_valid_frames"]
    if not isinstance(mvf, int) or mvf < 1:
        errors.append(f"min_valid_frames는 1 이상의 정수여야 합니다: {mvf!r}")

    return errors


# ─────────────────────────────────────────────────────────────
# 공개 함수
# ─────────────────────────────────────────────────────────────
def load_model_bundle(models_dir: Path | str = MODELS_DIR) -> ModelBundle:
    """세 산출물을 로드·검증해 ModelBundle로 반환한다. 문제가 있으면 ModelSpecError."""
    models_dir = Path(models_dir)
    paths = {name: models_dir / name for name in (MODEL_FILE, FEATURE_COLUMNS_FILE, TRAINING_INFO_FILE)}

    missing = [name for name, p in paths.items() if not p.is_file()]
    if missing:
        raise ModelSpecError(f"모델 산출물이 없습니다 ({models_dir}): {missing}")

    model = _load_model(paths[MODEL_FILE])
    feature_columns = _read_json(paths[FEATURE_COLUMNS_FILE])
    info = _read_json(paths[TRAINING_INFO_FILE])

    errors = _validate(model, feature_columns, info)
    if errors:
        raise ModelSpecError("모델 규격 검증 실패:\n  - " + "\n  - ".join(errors))

    bundle = ModelBundle(
        model=model,
        feature_columns=tuple(feature_columns),
        classes=tuple(str(c) for c in model.classes_),   # 반드시 model.classes_ 기준
        sequence_steps=int(info["sequence_steps"]),
        features_per_step=FEATURES_PER_STEP,
        min_valid_frames=int(info["min_valid_frames"]),
        expects_dataframe=hasattr(model, "feature_names_in_"),
        training_info=info,
    )
    logger.info("모델 로드 완료: %s", bundle.summary())
    return bundle


# ─────────────────────────────────────────────────────────────
# T01: 직접 실행 시 검증 결과 출력
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        b = load_model_bundle()
    except ModelSpecError as e:
        print(f"[T01 FAIL] {e}")
        sys.exit(1)

    s = b.summary()
    print("[T01 PASS] 모델 산출물 3종 로드·교차 검증 완료")
    print(f"  model             : {s['model_type']} (scikit-learn {s['sklearn_version']})")
    print(f"  input features    : {s['n_features']} = {s['sequence_steps']} steps × {s['features_per_step']}")
    print(f"  min_valid_frames  : {s['min_valid_frames']}")
    print(f"  expects DataFrame : {b.expects_dataframe}")
    print("  classes (predict_proba 열 순서):")
    for i, c in enumerate(s["classes"]):
        print(f"    [{i}] {c['action_id']}  {c['label']}")
