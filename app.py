import os
import json
import fitz  # PyMuPDF
import openai
import random
import xml.etree.ElementTree as ET
import requests
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
GG_DATA_PATH = "gg_employment_cached.json"

with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    static_companies = json.load(f)

gg_companies = []
if os.path.exists(GG_DATA_PATH):
    with open(GG_DATA_PATH, "r", encoding="utf-8") as f:
        gg_companies = json.load(f)

all_companies = static_companies + gg_companies

user_states = {}
shown_indices = {}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()

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
        print(f"❌ GPT 키워드 추출 오류: {e}")
        return []

def get_summary(c):
    return c.get("summary") or c.get("채용공고명") or ""

def tfidf_similarity(user_text, companies):
    documents = [user_text] + [get_summary(c) for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    if sims.max() > 0:
        sims = sims / sims.max()
    return sorted(zip(companies, sims), key=lambda x: x[1], reverse=True)

def generate_company_description(user_text, company, score):
    prompt = f"""
    당신은 채용 컨설턴트입니다. 아래 자기소개서를 분석하여 다음 기업의 업무와 얼마나 잘 맞는지 설명해 주세요.
    - 기업명: {company.get('회사명') or company.get('name')}
    - 업무: {company.get('채용공고명') or company.get('summary')}
    - 유사도 점수: {round(score, 2)}

    자기소개서:
    {user_text}
    """
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        desc = res.choices[0].message.content.strip()
        return desc
    except Exception as e:
        return f"설명 생성 중 오류 발생: {e}"

def ask_preferences():
    return "감사합니다. 관심 분야, 희망 근무지, 연봉을 입력해 주세요. 예시: 품질, 서울, 3000만원"

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
            state = {"user_text": user_text, "asked": False}
            user_states[user_id] = state
            return jsonify({"reply": ask_preferences()})

        if "user_text" not in state and message:
            state = {"user_text": message, "asked": False}
            user_states[user_id] = state
            return jsonify({"reply": ask_preferences()})

        if not state.get("asked") and message:
            parts = [p.strip().replace("만원", "") for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""
            state["asked"] = True
            user_states[user_id] = state

        if state.get("asked") and "user_text" in state:
            keywords = extract_keywords(state["user_text"])
            matched = tfidf_similarity(state["user_text"], all_companies)
            matched = [(c, s) for c, s in matched if s > 0.0]

            if user_id not in shown_indices:
                shown_indices[user_id] = 0

            start = shown_indices[user_id]
            count = 1 if "더 추천" in message else 3
            end = start + count
            selected = matched[start:end]
            shown_indices[user_id] += count

            if not selected:
                return jsonify({"reply": "더 이상 추천할 기업이 없습니다."})

            explanations = []
            for company, score in selected:
                desc = generate_company_description(state["user_text"], company, score)
                explanations.append(f"기업명: {company.get('회사명', '')}\n업무: {company.get('채용공고명', '')}\n유사도 점수: {round(score, 2)}\n설명: {desc}\n")

            return jsonify({"reply": "\n\n".join(explanations) + "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"})

        return jsonify({"reply": ask_preferences()})

    except Exception as e:
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
