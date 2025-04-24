from flask import Flask, request, jsonify, render_template
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
print("🔑 API 키 로드됨:", os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

assistant = client.beta.assistants.create(
    name="기업 추천 챗봇",
    instructions=system_prompt,
    model="gpt-4-1106-preview"
)

assistant_id = "asst_u2QSs359lwwE4ChWE9PO7p3K"  # ← Assistant ID 입력

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json["message"]

        # ✅ 매 요청마다 새 thread 생성
        thread = client.beta.threads.create()

        # 메시지 추가
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )

        # Run 실행
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        print("🚀 Run 생성:", run)

        # Run 완료까지 대기
        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print("⏳ Run 상태:", run.status)

        print("✅ 최종 Run 상태:", run.status)

        # 응답 메시지 조회
        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
        print("📥 메시지 목록:", messages)

        for msg in messages.data:
            print("🧾 역할:", msg.role)
            print("📦 콘텐츠:", msg.content)
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == "text":
                        print("🟢 GPT 응답:", content.text.value)
                        return jsonify(reply=content.text.value)

        return jsonify(reply="GPT 응답 없음")

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
