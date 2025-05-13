from flask import Flask, request, jsonify, render_template
import time
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)

assistant_id = os.getenv("ASSISTANT_ID")  # .env에 저장된 assistant ID
api_key = os.getenv("SERVICE_KEY")  # 기업정보 API Key

# 기업 검색 API 호출 함수
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
            data = response.json()
            items = data.get("response", {}).get("body", {}).get("items", [])
            if items:
                return f"기업명: {items[0].get('corpNm')} / 기업ID: {items[0].get('corpId')}"
            else:
                return "해당 기업에 대한 정보를 찾을 수 없습니다."
        else:
            return f"API 오류 발생 (code {response.status_code})"
    except Exception as e:
        return f"API 호출 중 예외 발생: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("file")
        file_content = ""

        # 파일이 첨부된 경우 내용 일부 추출
        if uploaded_file:
            print(f"📁 파일 업로드됨: {uploaded_file.filename}")
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            user_message += f"\n\n[첨부 파일 내용 일부]:\n{file_content[:1000]}"

        # 채용공고 요청이 있는 경우 자동 API 검색
        if "채용공고" in user_message:
            # 간단한 기업명 추출 예시 (실제 로직은 개선 가능)
            for word in user_message.split():
                if word.endswith("채용공고"):
                    corp_name = word.replace("채용공고", "").strip()
                    if corp_name:
                        corp_info = search_corporation(corp_name)
                        user_message += f"\n\n[기업 검색 결과]\n{corp_info}"
                    break

        # GPT Assistant와 대화 시작
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

        # 응답 대기
        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

        # 메시지 수신
        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="desc"
        )

        for msg in messages.data:
            for content in msg.content:
                if content.type == "text":
                    return jsonify(reply=content.text.value)

        return jsonify(reply="GPT 응답을 받지 못했습니다.")

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
