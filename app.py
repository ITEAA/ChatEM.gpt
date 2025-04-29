from flask import Flask, request, jsonify, render_template
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

app = Flask(__name__)

assistant_id = "asst_WW6opruOKAP1tdK7NAHvD6Yk"  # 실제 assistant ID 사용

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            print(f"📁 파일 업로드됨: {uploaded_file.filename}")
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            user_message += f"\n\n[첨부 파일 내용 요약]:\n{file_content[:1000]}"

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

        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")

        for msg in messages.data:
            for content in msg.content:
                if content.type == "text":
                    return jsonify(reply=content.text.value)

        return jsonify(reply="GPT 응답 없음")

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
