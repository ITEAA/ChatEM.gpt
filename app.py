import os
import fitz  # PyMuPDF
import openai
import json
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
PROXY_SERVER_URL = os.getenv("PROXY_SERVER_URL") or "http://localhost:8000"

# 1. 루트 엔드포인트 for Fly.io health check
@app.route("/")
def index():
    return "OK", 200

# 2. PDF or text 전처리

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# 3. GPT로 키워드 추출

def extract_keywords(text):
    prompt = f"""
다음 자기소개서에서 핵심 키워드를 5~7개 정도 추출해줘. 키워드만 JSON 배열로 반환해줘.

자기소개서:
{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    keywords = json.loads(response.choices[0].message.content)
    return keywords

# 4. KOSME 더미 프록시 서버 검색

def search_companies(keywords):
    query = " ".join(keywords)
    response = requests.get(
        f"{PROXY_SERVER_URL}/corp",
        params={
            "corpNm": query,
            "pageNo": 1,
            "numOfRows": 10,
            "resultType": "json"
        },
        timeout=10
    )
    if response.ok:
        return response.json().get("items", [])
    else:
        return []

# 5. 채팅 라우트
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message")
    file = request.files.get("file")

    if not user_message and not file:
        return jsonify({"reply": "❌ 자소서나 메시지를 입력해주세요."})

    try:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("uploads", filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)
            user_text = extract_text_from_pdf(filepath)
        else:
            user_text = user_message.strip()

        keywords = extract_keywords(user_text)
        companies = search_companies(keywords)

        if not companies:
            return jsonify({"reply": f"❌ 관련 기업 정보를 찾지 못했어요. (키워드: {keywords})"})

        reply = f"🔍 추출된 키워드: {keywords}\n\n추천 기업:\n"
        for c in companies:
            reply += f"- {c.get('corpNm', '기업명 없음')} ({c.get('adres', '주소 없음')})\n"

        return jsonify({"reply": reply.strip()})

    except Exception as e:
        return jsonify({"reply": f"❌ 오류가 발생했습니다: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
