# Django 웹 서버와 Celery Worker가 함께 쓰는 이미지
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/media

EXPOSE 8000
# 실제 실행 명령은 compose.yaml에서 서비스별로 지정 (django: runserver / celery: worker)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
