import os
import json
import fitz
import openai
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 기업 데이터 로드
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

# ✅ PDF 텍스트 추출 함수
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ GPT 키워드 추출
def extract_keywords(text):
    prompt = f"""
다음은 이력서 또는 자기소개서입니다. 핵심 키워드 5~7개를 쉼표로 구분하여 출력해 주세요.
--- 내용 ---
{text}
--- 키워드 ---
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"].strip()

# ✅ TF-IDF 유사도 계산
def calculate_similarity(user_text, companies):
    corpus = [user_text] + [f"{c['name']} {c['summary']}" for c in companies]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    for i, score in enumerate(scores):
        companies[i]["score"] = round(score, 2)
    companies.sort(key=lambda x: x["score"], reverse=True)
    return companies

# ✅ 추천 설명 생성 (분석가 시점)
def generate_analysis_recommendation(company, user_text):
    prompt = f"""
너는 채용 전문가로서 자기소개서와 기업 정보를 분석해, 해당 사용자가 왜 이 기업과 직무에 적합한지 논리적으로 분석한 추천 설명을 작성해야 한다.

[자기소개서 요약]
{user_text}

[기업명]
{company['name']}

[모집 직무]
{company['summary']}

위 내용을 기반으로 5~7문장으로 설명문을 작성해줘. 문체는 분석가 시점으로, '이 사용자는~', '자기소개서에 따르면~', '따라서 이 기업의 직무는~' 등의 표현을 써줘.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.6,
    )
    return response["choices"][0]["message"]["content"].strip()

# ✅ 추천 API
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.form
    file = request.files.get("file")
    text_input = data.get("text", "")
    mode = data.get("mode", "initial")

    user_text = extract_text_from_pdf(file) if file else text_input
    keywords = extract_keywords(user_text)
    print(f"🎯 추출 키워드: {keywords}")

    companies = calculate_similarity(user_text, company_data.copy())

    if mode == "initial":
        selected = companies[:3]
    elif mode == "more":
        selected = [companies[3]]
    else:
        return jsonify({"error": "invalid mode"}), 400

    results = []
    for company in selected:
        explanation = generate_analysis_recommendation(company, user_text)
        results.append({
            "기업명": company["name"],
            "업무": company["summary"],
            "유사도 점수": company["score"],
            "설명": explanation,
        })

    return jsonify({
        "추천 기업": results,
        "📌 안내": "더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
