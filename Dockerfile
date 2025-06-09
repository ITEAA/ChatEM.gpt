FROM python:3.10-slim

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 포트 설정 및 앱 실행
EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
