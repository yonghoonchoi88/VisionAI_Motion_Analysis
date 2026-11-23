"""
FastAPI AI 추론 서버 진입점 (main.py)

실행 (ai_service/ 폴더에서)
    uvicorn main:app --host 0.0.0.0 --port 8001
    → http://localhost:8001/docs 에서 API 문서·테스트

시작 시 1회
    1) 6일차 모델 3종 로드·검증 (services/model_loader.py)
    2) YOLO Pose 가중치 로드 + Warm-up (services/pose_pipeline.py)
    → app.state에 보관하고 모든 요청에서 재사용한다.
    로드에 실패해도 서버는 뜨며, /health가 원인을 알려 주고 /predict/video는 503을 반환한다.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI

from routers import predictions
from services.model_loader import ModelBundle, load_model_bundle
from services.pose_pipeline import PipelineConfig, PosePipeline

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_service")

Loader = Callable[[], tuple[ModelBundle, Any]]


def load_resources() -> tuple[ModelBundle, PosePipeline]:
    """운영용 로더: 실제 모델·YOLO를 불러온다."""
    bundle = load_model_bundle()
    pipeline = PosePipeline(
        PipelineConfig.load(),
        sequence_steps=bundle.sequence_steps,
        min_valid_frames=bundle.min_valid_frames,
    )
    if os.getenv("SKIP_WARMUP", "0") != "1":
        pipeline.warmup()
    return bundle, pipeline


def create_app(loader: Loader = load_resources) -> FastAPI:
    """loader를 바꿔 끼우면 테스트에서 가짜 모델로 같은 앱을 띄울 수 있다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        app.state.pipeline = None
        app.state.load_error = None
        try:
            app.state.bundle, app.state.pipeline = loader()
            logger.info("AI 추론 서버 준비 완료 (device=%s)", app.state.pipeline.device)
        except Exception as e:  # 서버는 띄우고 /health로 원인을 보고한다
            app.state.load_error = f"{type(e).__name__}: {e}"
            logger.exception("모델 로드 실패 — /predict/video는 503을 반환합니다.")
        yield

    app = FastAPI(
        title="VisionAI Motion Analysis — AI Inference Service",
        description="6일차 Random Forest + YOLO Pose/ByteTrack으로 영상 속 사람별 동작(6종)을 예측합니다.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(predictions.router)
    return app


app = create_app()
