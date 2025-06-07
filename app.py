import os
import json
import pickle
import difflib
import random
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load company data only once (lightweight)
with open("ChatEM_companies_top1000.json", "r", encoding="utf-8") as f:
    companies = json.load(f)

def load_vectorizer_and_matrix():
    with open("ChatEM_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("ChatEM_tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return vectorizer, tfidf_matrix

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_keywords_from_text(text):
    # 단순 키워드 추출 (TF-IDF 기반)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = tfidf.fit_transform([text])
    return tfidf.get_feature_names_out()

def recommend_companies(user_text, user_field, user_location, user_salary):
    vectorizer, tfidf_matrix = load_vectorizer_and_matrix()
    user_vec = vectorizer.transform([user_text])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    company_scores = []
    for idx, score in enumerate(cosine_similarities):
        company = companies[idx]
        if (user_field.lower() in company['분야'].lower() and
            user_location in company['지역'] and
            int(company['연봉']) <= int(user_salary)):
            company_scores.append((company, score))

    # 점수 기준 정렬 및 상위 2~3개 추천
    company_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = company_scores[:3] if company_scores else []

    # 추천 결과 구성
    results = []
    for company, score in top_matches:
        results.append({
            "회사명": company['회사명'],
            "지역": company['지역'],
            "분야": company['분야'],
            "연봉": company['연봉'],
            "유사도점수": f"{score*100:.2f}%",
            "설명": f"'{company['회사명']}'은(는) 사용자의 관심 분야({user_field}) 및 지역({user_location})과 잘 부합하며, 연봉 조건도 충족합니다."
        })
    return results

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.form
        user_field = data.get("field", "")
        user_location = data.get("location", "")
        user_salary = data.get("salary", "0")

        # 파일 또는 텍스트 입력 처리
        if 'file' in request.files:
            file = request.files['file']
            user_text = extract_text_from_pdf(file)
        else:
            user_text = data.get("text", "")

        if not user_text.strip():
            return jsonify({"error": "이력서나 자기소개서를 입력하거나 업로드하세요."}), 400

        recommendations = recommend_companies(user_text, user_field, user_location, user_salary)
        return jsonify({"추천결과": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
