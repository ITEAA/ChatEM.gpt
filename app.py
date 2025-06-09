import os
import json
import fitz  # PyMuPDF
import openai
import requests
import xml.etree.ElementTree as ET
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"

# Load top20 companies and cached 경기 채용공고 데이터
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    company_data_top20 = json.load(f)

with open("gg_employment_cached.json", "r", encoding="utf-8") as f:
    company_data_gg = json.load(f)

company_data = company_data_top20 + company_data_gg
user_states = {}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text.strip()

def extract_keywords(text):
    prompt = f"""
    다음 자기소개서 또는 이력서에서 핵심 키워드를 추출해줘.
    - 5~10개 정도 뽑아줘.
    - 키워드는 콤마(,)로 구분해서 출력해줘.

    내용:
    {text}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        result = response.choices[0].message.content
        return [kw.strip() for kw in result.split(",") if kw.strip()]
    except Exception as e:
        print(f"❌ GPT 호출 에러: {e}")
        return []

def filter_companies(keywords, interest, region, salary):
    def score(company):
        base_score = 0
        if any(kw in (company.get("summary") or "") for kw in keywords):
            base_score += 1
        if interest and interest in (company.get("summary") or ""):
            base_score += 0.3
        if region and region in (company.get("region") or company.get("근무지역") or ""):
            base_score += 0.3
        if salary and str(salary) in (company.get("salary") or company.get("급여") or ""):
            base_score += 0.2
        return base_score

    return sorted(company_data, key=score, reverse=True)

def tfidf_similarity(user_text, companies):
    def get_summary(company):
        return company.get("summary") or company.get("채용공고명") or ""

    documents = [user_text] + [get_summary(c) for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sorted(zip(companies, cosine_sim), key=lambda x: x[1], reverse=True)

def generate_reason(user_text, companies_with_scores):
    companies_info = []
    for company, score in companies_with_scores:
        companies_info.append({
            "name": company.get("회사명") or company.get("name"),
            "summary": company.get("summary") or company.get("채용공고명"),
            "score": round(score, 2),
            "url": company.get("url") or company.get("채용정보URL")
        })

    prompt = f"""
당신은 채용 컨설턴트입니다.
다음 자기소개서 내용을 바탕으로, 각 기업이 사용자의 경력과 얼마나 잘 맞는지 분석해 주세요.
출력 형식:

기업명: OOO
업무: OOO
유사도 점수: 0.XX
설명: (분석가 시점에서 자기소개서 경험과 해당 직무의 연결성 분석)

[자기소개서 내용]
{user_text}

[기업 목록 및 유사도 점수]
{json.dumps(companies_info, ensure_ascii=False)}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        explanation = response.choices[0].message.content
        explanation += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
        return explanation
    except Exception as e:
        print(f"❌ GPT 추천 설명 생성 에러: {e}")
        return "추천 이유를 생성하는 중 오류가 발생했습니다."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.remote_addr
    message = request.form.get("message", "").strip()
    file = request.files.get("file")
    state = user_states.get(user_id, {})

    try:
        if file:
            user_text = extract_text_from_pdf(file)
            state["user_text"] = user_text

        if message and "," in message and "만원" in message:
            parts = [p.strip().replace("만원", "") for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""

        if message and "user_text" not in state:
            state["user_text"] = message

        if "user_text" in state:
            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, state.get("interest"), state.get("region"), state.get("salary"))
            matched = tfidf_similarity(state["user_text"], filtered)
            selected = matched[:3] if "더 추천해줘" not in message else [matched[3]]
            explanation = generate_reason(state["user_text"], selected)
            return jsonify({"reply": explanation})

        missing = []
        if "user_text" not in state:
            missing.append("자기소개서 또는 이력서")
        if missing:
            return jsonify({"reply": f"먼저 {', '.join(missing)}를 입력해 주세요."})

        return jsonify({"reply": "입력을 인식하지 못했습니다. 다시 시도해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
