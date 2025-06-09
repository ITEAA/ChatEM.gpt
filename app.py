import os
import json
import difflib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder="templates")
CORS(app)

# 기업 데이터 로드
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

# 홈페이지 라우팅
@app.route("/")
def index():
    return render_template("index.html")

# 채팅 처리
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_text = request.json.get("message", "")

        if not user_text.strip():
            return jsonify({"error": "빈 입력입니다."})

        # 키워드 추출
        keywords = extract_keywords(user_text)

        # 기업 추천
        result = recommend_companies_by_similarity(keywords)

        return jsonify({"message": result})
    except Exception as e:
        return jsonify({"error": f"❌ 오류 발생: {str(e)}"}), 500

# 키워드 추출 함수 (GPT 사용)
def extract_keywords(text):
    prompt = f"""다음 자기소개서 또는 이력서에서 핵심 키워드를 5~10개 추출해줘. 쉼표로 구분해서 출력해.
예시: 인공지능, 데이터분석, 파이썬, 문제해결능력, 커뮤니케이션
내용: {text}
결과:"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    keyword_text = response.choices[0].message.content.strip()
    return keyword_text.replace("\n", " ").replace("결과:", "").strip()

# 유사도 기반 추천
def recommend_companies_by_similarity(user_keywords):
    user_keywords = user_keywords.lower()
    company_names = []
    tag_texts = []

    for company in company_data:
        company_names.append(company["회사명"])
        tag_texts.append(" ".join(company.get("태그", [])).lower())

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([user_keywords] + tag_texts)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    top_indices = cosine_scores.argsort()[-2:][::-1]
    recommendations = []
    for idx in top_indices:
        company = company_data[idx]
        score = round(cosine_scores[idx] * 100, 2)
        recommendations.append(f"✅ {company['회사명']} (유사도: {score}%) - {company['요약설명']}")

    return "\n".join(recommendations)

# 로컬 실행용 (Fly.io는 gunicorn 사용)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
