import os
import json
import fitz  # PyMuPDF
import openai
import xml.etree.ElementTree as ET
import requests
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
GG_API_KEY = "8af0f404ca144249be0cfab9728b619b"

user_states = {}

# Load company data from both static file and cached API result
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    static_company_data = json.load(f)

with open("gg_employment_cached.json", "r", encoding="utf-8") as f:
    gg_company_data = json.load(f)

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

def filter_companies(companies, keywords, interest, region, salary):
    def score(company):
        base_score = 0
        summary = company.get("summary") or ""
        if any(kw in summary for kw in keywords):
            base_score += 1
        if interest and interest in summary:
            base_score += 0.3
        if region and region in company.get("근무지역", ""):
            base_score += 0.3
        if salary and str(salary) in company.get("급여", ""):
            base_score += 0.2
        return base_score

    return sorted(companies, key=score, reverse=True)

def tfidf_similarity(user_text, companies):
    documents = [user_text] + [(c.get("summary") or c.get("채용공고명") or "") for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sorted(zip(companies, cosine_sim), key=lambda x: x[1], reverse=True)

def generate_reason(user_text, companies_with_scores, interest, region, salary):
    explanations = []
    for company, score in companies_with_scores:
        prompt = f"""
당신은 채용 컨설턴트입니다.
아래 자기소개서와 사용자 조건을 참고하여 다음 기업에 왜 적합한지 분석가 시점으로 설명해 주세요.

기업명: {company.get("회사명") or company.get("name")}
업무: {company.get("채용공고명") or company.get("summary")}
유사도 점수: {round(score, 3)}

[자기소개서]
{user_text}

[사용자 입력 조건]
관심 분야: {interest or "(입력 안됨)"}
희망 근무지: {region or "(입력 안됨)"}
희망 연봉: {salary or "(입력 안됨)"}
"""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            explanations.append(response.choices[0].message.content.strip())
        except Exception as e:
            explanations.append(f"❌ 설명 생성 오류: {e}")
    return "\n\n".join(explanations) + "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"

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
            state["user_text"] = extract_text_from_pdf(file)

        if message and "," in message and "만원" in message:
            parts = [p.strip().replace("만원", "") for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""

        if message and "user_text" not in state:
            state["user_text"] = message

        if "user_text" in state:
            keywords = extract_keywords(state["user_text"])
            combined_data = static_company_data + gg_company_data
            filtered = filter_companies(combined_data, keywords, state.get("interest"), state.get("region"), state.get("salary"))
            matched = tfidf_similarity(state["user_text"], filtered)
            selected = matched[:3] if "더 추천해줘" not in message else [matched[3]]
            explanation = generate_reason(state["user_text"], selected, state.get("interest"), state.get("region"), state.get("salary"))
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
