import time
import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ✅ 명시적으로 API 키 사용
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Assistant 객체는 생성된 ID를 직접 지정하거나 1회만 생성되게 해야 함
assistant = client.beta.assistants.create(
    name="웹 챗봇",
    instructions="친절하고 유익하게 응답하세요.",
    model="gpt-4-1106-preview"
)
thread = client.beta.threads.create()

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status not in ["completed", "failed", "cancelled"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
    for msg in messages.data:
        for content in msg.content:
            if content.type == "text":
                return jsonify(reply=content.text.value)

    return jsonify(reply="죄송합니다. 응답을 가져오지 못했습니다.")
