from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

app = Flask(__name__)
CORS(app)

# 더미 기업 데이터 (실제로는 DB 또는 API로부터 불러올 것)
dummy_companies = [
    {"name": "에이아이로직스", "description": "AI 기반 물류 최적화 솔루션 개발", "location": "성남", "salary": "3000"},
    {"name": "그로스랩", "description": "데이터 기반 마케팅 자동화 플랫폼", "location": "수원", "salary": "3200"},
    {"name": "메타인텔리전스", "description": "생성형 AI 기술 연구 및 서비스 제공", "location": "용인", "salary": "3500"},
]

@app.route("/")
def home():
    return "✅ Flask 서버 로컬 실행 중입니다."

@app.route("/api/test")
def test():
    return jsonify({"message": "서버 연결 성공", "status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "PDF 파일만 지원됩니다."}), 400

    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    return jsonify({"text": text.strip()})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "자기소개서 텍스트가 없습니다."}), 400

    user_text = data["text"]

    # 기업 설명 + 사용자 자기소개서 텍스트 TF-IDF 유사도 계산
    corpus = [user_text] + [c["description"] for c in dummy_companies]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    top_indices = similarities.argsort()[::-1][:2]  # 유사도 상위 2개
    results = []
    for idx in top_indices:
        company = dummy_companies[idx]
        score = round(float(similarities[idx]) * 100, 2)
        results.append({
            "company": company["name"],
            "description": company["description"],
            "location": company["location"],
            "salary": company["salary"],
            "similarity": score,
            "reason": f"당신의 자기소개서와 '{company['name']}' 기업의 특성이 {score}% 유사합니다."
        })

    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
