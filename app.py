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

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-api-key"
user_states = {}

# 기업 데이터 불러오기
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    top20_companies = json.load(f)

with open("gg_employment_cached.json", "r", encoding="utf-8") as f:
    gg_companies = json.load(f)

all_companies = top20_companies + gg_companies

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()

def extract_keywords(text):
    prompt = f"""
다음 자기소개서에서 핵심 키워드를 5~10개 추출해줘. 키워드는 콤마(,)로 구분해서 출력해줘.

자기소개서:
{text}
"""
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        result = res.choices[0].message.content
        return [kw.strip() for kw in result.split(",") if kw.strip()]
    except Exception as e:
        print(f"❌ GPT 키워드 추출 오류: {e}")
        return []

def tfidf_similarity(user_text, companies, interest, region, salary):
    def get_text(company):
        return company.get("summary") or company.get("채용공고명") or ""

    documents = [user_text] + [get_text(c) for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    sim_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    scored = []
    for i, company in enumerate(companies):
        score = sim_scores[i]
        if score == 0.0:
            continue  # 유사도 0.0 기업 제외

        # 조건 보정 (작게 반영)
        bonus = 0
        if interest and interest in (company.get("summary") or ""):
            bonus += 0.05
        if region and region in (company.get("region") or company.get("시군명") or ""):
            bonus += 0.05
        if salary and str(salary) in (company.get("salary") or ""):
            bonus += 0.03

        scored.append((company, score + bonus))

    return sorted(scored, key=lambda x: x[1], reverse=True)

def generate_individual_reason(user_text, companies_with_scores):
    messages = []
    for company, score in companies_with_scores:
        company_name = company.get("회사명") or company.get("name")
        summary = company.get("summary") or company.get("채용공고명")
        score = round(score, 2)

        prompt = f"""
당신은 채용 컨설턴트입니다. 아래 기업이 사용자의 자기소개서와 얼마나 잘 맞는지 분석가 시점에서 설명해 주세요.

형식:
기업명: {company_name}
업무: {summary}
유사도 점수: {score}
설명: (사용자의 자기소개서 경험과 해당 직무의 연결점을 중심으로 구체적이고 설득력 있게 설명)

[자기소개서]
{user_text}
"""
        try:
            res = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            explanation = res.choices[0].message.content.strip()
            messages.append(explanation)
        except Exception as e:
            print(f"❌ GPT 기업 설명 오류: {company_name} - {e}")
            continue

    final = "\n\n".join(messages)
    final += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
    return final

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
            user_states[user_id] = state
            return jsonify({"reply": "자기소개서를 잘 받았습니다. 관심 분야, 희망 근무지, 연봉을 입력해 주세요. 예시: 품질, 서울, 3000만원"})

        if message and "," in message and "만원" in message:
            parts = [p.strip().replace("만원", "") for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""
            user_states[user_id] = state

        elif message and "더 추천해줘" in message:
            if "user_text" in state and "ranked_companies" in state:
                prev = set(state.get("recommended_ids", []))
                remaining = [c for c in state["ranked_companies"] if id(c[0]) not in prev]
                if not remaining:
                    return jsonify({"reply": "더 이상 추천할 기업이 없습니다."})
                selected = remaining[0:1]
                state.setdefault("recommended_ids", []).extend(id(c[0]) for c in selected)
                explanation = generate_individual_reason(state["user_text"], selected)
                return jsonify({"reply": explanation})

        elif message and "user_text" not in state:
            state["user_text"] = message
            user_states[user_id] = state
            return jsonify({"reply": "감사합니다. 관심 분야, 희망 근무지, 연봉을 입력해 주세요. 예시: 품질, 서울, 3000만원"})

        # 추천
        if "user_text" in state and all(k in state for k in ("interest", "region", "salary")):
            keywords = extract_keywords(state["user_text"])
            matched = tfidf_similarity(state["user_text"], all_companies, state["interest"], state["region"], state["salary"])
            selected = matched[:3]
            state["ranked_companies"] = matched
            state["recommended_ids"] = [id(c[0]) for c in selected]
            user_states[user_id] = state
            explanation = generate_individual_reason(state["user_text"], selected)
            return jsonify({"reply": explanation})

        return jsonify({"reply": "입력을 인식하지 못했습니다. 다시 시도해 주세요."})

    except Exception as e:
        print(f"❌ 서버 오류: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
