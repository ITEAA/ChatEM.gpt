FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# 미리 PyTorch 설치
RUN pip install --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 나머지 requirements 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
