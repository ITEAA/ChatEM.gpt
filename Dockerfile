# 1. Slim Python 베이스 이미지 사용
FROM python:3.10-slim

# 2. 환경 변수 설정 (필요시)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. 필수 패키지 설치 (PyMuPDF 등 빌드에 필요한 라이브러리 포함)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    gcc \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리 생성 및 이동
WORKDIR /app

# 5. 종속성 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 코드 복사
COPY . .

# 7. 기본 포트 환경변수 설정 (Fly.io 등에서 8080 사용)
ENV PORT=8080

# 8. 실행 명령어 설정 (gunicorn 사용 권장)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
