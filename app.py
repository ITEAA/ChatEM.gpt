from flask import Flask, request, jsonify, render_template
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

app = Flask(__name__)

assistant_id = "asst_WW6opruOKAP1tdK7NAHvD6Yk"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("file")

        # ✅ 파일이 첨부되었을 경우, 텍스트 내용 일부를 메시지에 포함
        if uploaded_file and uploaded_file.filename:
            print(f"📁 업로드된 파일명: {uploaded_file.filename}")
            try:
                file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                user_message += f"\n\n[첨부 파일 내용 발췌]:\n{file_content[:1000]}"  # 최대 1000자
            except Exception as file_err:
                print(f"⚠️ 파일 처리 오류: {file_err}")

        # ✅ 쓰레드 생성 → 메시지 추가 → Run 실행
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message or "안녕하세요!"
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
