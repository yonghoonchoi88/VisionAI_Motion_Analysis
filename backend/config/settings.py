"""
Django 설정 (config/settings.py)

모든 비밀값·환경별 값은 환경변수로 받는다 (.env → docker compose가 주입, 로컬은 export 또는 .env 로드).
    DB_ENGINE=postgres(기본) | sqlite     ← sqlite는 테스트·빠른 로컬 확인용
    CELERY_TASK_ALWAYS_EAGER=1            ← Redis 없이 요청 안에서 즉시 실행 (개발용)
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    return env(key, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def env_list(key: str, default: str = "") -> list[str]:
    return [v.strip() for v in env(key, default).split(",") if v.strip()]


# 로컬 실행 시 저장소 루트의 .env를 읽는다 (Docker에서는 compose가 이미 주입하므로 덮어쓰지 않음)
_dotenv = BASE_DIR.parent / ".env"
if _dotenv.is_file():
    for line in _dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ─────────────────────────────────────────────────────────────
# 보안·기본
# ─────────────────────────────────────────────────────────────
SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "accounts",
    "analyses",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ─────────────────────────────────────────────────────────────
# DB: PostgreSQL (서비스) / SQLite (테스트·간이 확인)
# ─────────────────────────────────────────────────────────────
if env("DB_ENGINE", "postgres") == "sqlite":
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("POSTGRES_DB", "vision_action"),
            "USER": env("POSTGRES_USER", "vision"),
            "PASSWORD": env("POSTGRES_PASSWORD", "vision"),
            "HOST": env("POSTGRES_HOST", "localhost"),
            "PORT": env("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

# ─────────────────────────────────────────────────────────────
# 정적·미디어 파일 (업로드 영상은 DB가 아니라 MEDIA_ROOT에 저장, DB에는 경로만)
# ─────────────────────────────────────────────────────────────
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = Path(env("MEDIA_ROOT", str(BASE_DIR / "media")))

DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024        # 이보다 큰 업로드는 임시 파일로 받음

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "analyses:index"
LOGOUT_REDIRECT_URL = "login"

# ─────────────────────────────────────────────────────────────
# DRF: 브라우저 세션 로그인 기반 (JS fetch는 CSRF 토큰을 함께 보냄)
# ─────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.SessionAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# ─────────────────────────────────────────────────────────────
# Celery + Redis (Redis는 '작업 전달자', 영구 결과는 PostgreSQL)
# ─────────────────────────────────────────────────────────────
CELERY_BROKER_URL = env("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", False)
CELERY_TASK_EAGER_PROPAGATES = False
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True                  # Worker가 중간에 죽으면 작업을 다시 받음
CELERY_WORKER_PREFETCH_MULTIPLIER = 1         # 무거운 작업은 한 번에 하나씩
CELERY_TASK_TIME_LIMIT = 15 * 60
CELERY_TIMEZONE = TIME_ZONE

# ─────────────────────────────────────────────────────────────
# AI 서비스·입력 정책
# ─────────────────────────────────────────────────────────────
FASTAPI_URL = env("FASTAPI_URL", "http://localhost:8001").rstrip("/")
FASTAPI_TIMEOUT_SECONDS = float(env("FASTAPI_TIMEOUT_SECONDS", "300"))
MAX_UPLOAD_MB = int(env("MAX_UPLOAD_MB", "200"))
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# 허용 IP Webcam 목록 (host:port). 사용자가 임의 URL을 넣어 서버가 외부로 요청하는 것(SSRF)을 막는다.
ALLOWED_CAMERA_HOSTS = env_list("ALLOWED_CAMERA_HOSTS", "")
CAMERA_CAPTURE_SECONDS_DEFAULT = int(env("CAMERA_CAPTURE_SECONDS", "8"))
CAMERA_CAPTURE_SECONDS_MIN = 5
CAMERA_CAPTURE_SECONDS_MAX = 10

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "root": {"handlers": ["console"], "level": env("LOG_LEVEL", "INFO")},
}
