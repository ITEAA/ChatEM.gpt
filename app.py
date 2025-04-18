from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os

# 🔑 API 키 불러오기
load_dotenv()
print("🔑 API 키 로드됨:", os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # .env에서 로드

system_prompt = "당신은 친절하고 정확한 기업 추천 AI 챗봇입니다. 사용자의 관심사, 이력서, 자기소개서 내용을 바탕으로 유익하고 신뢰도 높은 기업을 추천해 주세요."

# ✅ 프롬프트 함수 분리
def make_messages(user_input):
    return [
        {"role": "system", "content" : system_prompt},
        {"role": "user", "content" :  user_input}
    ]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json["message"]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 gpt-4 (API 충전 상태라면)
            messages=make_messages(user_message)
        )

        reply = response.choices[0].message.content.strip()
        print("🟢 GPT 응답:", reply)
        return jsonify(reply=reply)

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
