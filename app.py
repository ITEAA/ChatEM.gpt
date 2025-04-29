from flask import Flask, request, jsonify, render_template
import time
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)

# 🛠 Assistant 생성 (functions 등록 포함)
functions = [
    {
        "name": "search_corporation",
        "description": "기업 이름으로 기업 정보를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "corpNm": {
                    "type": "string",
                    "description": "조회할 기업 이름 (예: 삼성전자)",
                },
            },
            "required": ["corpNm"],
        },
    }
]

with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

assistant = client.beta.assistants.create(
    name="기업 추천 챗봇",
    instructions=system_prompt,
    model="gpt-4-1106-preview",
    tools=[{"type": "function", "function": func} for func in functions]
)

assistant_id = assistant.id  # 새로 만든 assistant의 id 사용

# 🛠 기업 검색 함수
def search_corporation(corpNm):
    api_url = "https://corp-api-rho.vercel.app/corp"
    service_key = "ZqDMcB9z2xwM8pqNALpRI0Dy4jqugWQPfSBFwEWeOe6GXmHv/JOjl0xmZKTME66FX/SOUwK9vjShZ7ms04STmA=="
    params = {
        "pageNo": 1,
        "numOfRows": 10,
        "resultType": "json",
        "corpNm": corpNm,
        "serviceKey": service_key
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", [])
        if items:
            return f"🔎 기업 검색 결과: {items[0]['corpNm']} (ID: {items[0]['corpId']})"
        else:
            return "❌ 기업 정보를 찾을 수 없습니다."
    else:
        return "❌ API 호출 실패"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "")

        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Run 대기
        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs
