from flask import Flask, request, jsonify, render_template, session
import time
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = "your-secret-key"  # 세션 사용을 위해 필요

assistant_id = os.getenv("ASSISTANT_ID")
api_key = os.getenv("SERVICE_KEY")

# 기업 검색 함수 (간단한 예시)
def search_corporation(corp_name):
    try:
        url = "https://corp-api-rho.vercel.app/corp"
        params = {
            "pageNo": 1,
            "numOfRows": 1,
            "resultType": "json",
            "corpNm": corp_name,
            "serviceKey": api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get("response", {}).get("body", {}).get("items", [])
            if items:
                return f"기업명: {items[0]['corpNm']} / 기업ID: {items[0]['corpId']}"
            else:
                return "해당 기업 정보를 찾을 수 없습니다."
        else:
            return f"API 오류: {response.status_code}"
    except Exception as e:
        return f"API 호출 예외 발생: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("file")

        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            user_message += f"\n\n[첨부 파일 내용 요약]:\n{file_content[:1000]}"

        # 기업명 검색 요청 감지
        if "채용공고" in user_message:
            for word in user_message.split():
                if word.endswith("채용공고"):
                    corp_name = word.replace("채용공고", "").strip()
                    if corp_name:
                        corp_info = search_corporation(corp_name)
                        user_message += f"\n\n[기업 검색 결과]:\n{corp_info}"
                    break

        # 세션 thread_id 관리
        if "thread_id" not in session:
            thread = client.beta.threads.create()
            session["thread_id"] = thread.id
        else:
            thread = client.beta.threads.retrieve(session["thread_id"])

        # 메시지 보내기
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )

        # GPT Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # ⏳ 타임아웃 처리
        timeout = 30  # 최대 30초
        start_time = time.time()

        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if time.time() - start_time > timeout:
                return jsonify(reply="GPT 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요.")

        # 응답 메시지 추출
        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")

        for msg in messages.data:
            for content in msg.content:
                if content.type == "text":
                    return jsonify(reply=content.text.value)

        return jsonify(reply="GPT로부터 응답을 받지 못했습니다.")

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류가 발생했습니다: " + str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
