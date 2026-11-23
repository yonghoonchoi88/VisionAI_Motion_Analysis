# FastAPI AI 추론 서버 이미지 (CPU 전용 — 이미지 크기를 GPU판보다 수 GB 줄임)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POSE_DEVICE=cpu

# opencv-python(ultralytics 의존성)이 필요로 하는 시스템 라이브러리
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch는 CPU 전용 휠을 먼저 설치 → requirements.txt의 torch==2.15.0 조건을 이미 만족하므로 GPU판을 받지 않음
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision \
 && pip install -r requirements.txt

COPY . .
# models/ 폴더는 이미지에 넣지 않고 compose에서 읽기 전용 볼륨으로 연결한다 (.dockerignore 참고)

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
