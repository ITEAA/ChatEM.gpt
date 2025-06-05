from flask import Flask, request, jsonify, render_template
import os
import re
import requests
import xml.etree.ElementTree as ET
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
job_api_key = os.getenv("JOB_API_KEY")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        user_prefs = extract_user_preferences(user_input)

        # 실제 API 대신 더미 데이터 사용 (방화벽 미허용 대비)
        dummy_companies = [
            {"name": "경남IT솔루션", "tags": ["진주", "소프트웨어", "개발", "백엔드"]},
            {"name": "진주로직스", "tags": ["물류", "운송", "경상남도", "물류관리"]},
            {"name": "에코그린테크", "tags": ["환경", "에너지", "친환경", "진주"]},
            {"name": "네오교육", "tags": ["에듀테크", "교육", "콘텐츠", "웹"]},
        ]

        match = match_company_to_user(dummy_companies, keywords, user_prefs)

        if not match:
            return jsonify({"reply": "❌ 기업 정보를 불러오지 못했습니다. 나중에 다시 시도해주세요."})

        prompt = build_explanation_prompt(keywords, user_prefs, match)
        reply = get_gpt_reply(prompt)

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

def extract_resume_text(text):
    return text  # 간단히 원문 반환

def extract_keywords(text):
    prompt = f"다음 자기소개서에서 핵심 기술, 직무, 경험 키워드를 쉼표로 추출해줘:\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return [kw.strip() for kw in response.choices[0].message.content.split(",")]
    except Exception as e:
        print("❌ 키워드 추출 실패:", e)
        return ["개발", "팀워크", "문제해결"]

def extract_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]

def compute_similarity(text1, text2):
    try:
        emb1 = client.embeddings.create(input=text1, model="text-embedding-ada-002").data[0].embedding
        emb2 = client.embeddings.create(input=text2, model="text-embedding-ada-002").data[0].embedding
        return cosine_similarity(emb1, emb2)
    except Exception as e:
        print("❌ 유사도 계산 실패:", str(e))
        return 0.0

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)

def match_company_to_user(companies, keywords, prefs):
    user_text = " ".join(keywords + prefs)
    best = None
    best_score = -1
    for company in companies:
        company_text = " ".join(company["tags"])
        score = compute_similarity(user_text, company_text)
        if score > best_score:
            best = company
            best_score = score
    return best or (companies[0] if companies else None)

def build_explanation_prompt(keywords, preferences, company, job_summary=""):
    base = f"다음 사용자 정보와 추천 기업을 기반으로, 왜 이 기업이 적합한지 설명해주세요.\n\n"
    base += f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join(preferences)}\n\n"
    if company is None:
        base += "[추천 기업 정보 없음]\n- 기업 추천에 실패했습니다."
        return base
    base += f"[추천 기업]\n- 기업명: {company['name']}\n- 태그: {', '.join(company['tags'])}"
    if job_summary:
        base += f"\n\n[채용공고]\n{job_summary}"
    return base

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT 응답 실패:", e)
        return "GPT 응답 생성 실패"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
