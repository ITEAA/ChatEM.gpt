FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler libsentencepiece-dev && \
    pip install --no-cache-dir -r requirements.txt
    
COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]

COPY ChatEM_top20_companies.json ./
