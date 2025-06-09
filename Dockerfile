FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# PyTorch 2.1.2+cpu 설치 (transformers 호환 보장)
RUN pip install --no-cache-dir torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 나머지 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
