# run_local.py
from flask import Flask, request, jsonify
from app import app  # app.py에 app = Flask(__name__) 있어야 함
import os
import fitz  # PyMuPDF
import tempfile
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

# 샘플 기업 데이터 (실제로는 API 또는 DB에서 가져옴)
dummy_companies = [
    {"회사명": "에이아이코리아", "설명": "인공지능 기반 헬스케어 솔루션 개발", "키워드": "AI, 헬스케어, Python"},
    {"회사명": "지피티솔루션", "설명": "GPT 기반 챗봇 플랫폼 운영", "키워드": "GPT, 챗봇, 딥러닝"},
    {"회사명": "데이터브릿지", "설명": "데이터 분석 및 시각화 도구 개발", "키워드": "데이터, 분석, 시각화"},
]

@app.route("/")
def home():
    return "✅ 로컬 Flask 서버 정상 작동 중입니다."

@app.route("/api/test")
def test():
    return jsonify({"message": "테스트 성공", "status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "파일 없음"}), 400
    filename = secure_filename(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file.save(temp_file.name)
        pdf_text = extract_text_from_pdf(temp_file.name)
        os.unlink(temp_file.name)

    return jsonify({"filename": filename, "content": pdf_text[:300]})  # 앞 300자만 보여줌

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    if not data or "content" not in data:
        return jsonify({"error": "자기소개서 본문 없음"}), 400

    user_text = data["content"]
    scores = []

    for company in dummy_companies:
        company_text = company["키워드"] + " " + company["설명"]
        sim_score = SequenceMatcher(None, user_text, company_text).ratio()
        scores.append((sim_score, company))

    scores.sort(reverse=True, key=lambda x: x[0])
    top_matches = [
        {
            "회사명": comp["회사명"],
            "유사도": round(score, 2),
            "설명": comp["설명"]
        }
        for score, comp in scores[:2]
    ]

    return jsonify({"추천기업": top_matches})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
