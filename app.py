import os
import json
import fitz  # PyMuPDF
import openai
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
GG_CACHED_FILE = "gg_employment_cached.json"

user_states = {}

# Load GG cached data
with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
    cached_companies = json.load(f)

# 유사도 계산

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
        print(f"❌ GPT 키워드 추출 오류: {e}")
        return []

def tfidf_similarity(user_text, companies):
    def get_summary(company):
        return f"{company.get('채용공고명', '')} {company.get('회사명', '')}"

    documents = [user_text] + [get_summary(c) for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sorted(zip(companies, cosine_sim), key=lambda x: x[1], reverse=True)

def generate_reason_individual(user_text, company, score):
    prompt = f"""
당신은 채용 분석가입니다. 아래 자기소개서와 기업 정보를 참고하여, 이 사용자가 왜 이 기업의 해당 직무에 적합한지 설명해 주세요.

기업명: {company.get('회사명')}
업무: {company.get('채용공고명')}
유사도 점수: {round(score, 2)}

[자기소개서]
{user_text}

[설명]
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT 설명 오류: {e}")
        return "설명을 생성하는 데 문제가 발생했습니다."

def make_recommendations(user_text, interest, region, salary, shown=set(), top_n=3):
    keywords = extract_keywords(user_text)

    def score(company):
        s = 0
        summary = company.get("채용공고명", "") + company.get("회사명", "")
        if any(kw in summary for kw in keywords):
            s += 1
        if interest and interest in summary:
            s += 0.3
        if region and region in company.get("근무지역", ""):
            s += 0.3
        if salary and str(salary) in company.get("급여", ""):
            s += 0.2
        return s

    filtered = sorted(cached_companies, key=score, reverse=True)
    tfidf_ranked = tfidf_similarity(user_text, filtered)
    results = []
    for comp, sim in tfidf_ranked:
        if sim > 0.0 and (comp.get("회사명"), comp.get("채용공고명")) not in shown:
            shown.add((comp.get("회사명"), comp.get("채용공고명")))
            results.append((comp, sim))
        if len(results) >= top_n:
            break
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.remote_addr
    message = request.form.get("message", "").strip()
    file = request.files.get("file")
    state = user_states.get(user_id, {"shown": set()})

    try:
        # 1. PDF 업로드 or 첫 자소서 텍스트 입력
        if file:
            user_text = extract_text_from_pdf(file)
            state["user_text"] = user_text
            user_states[user_id] = state
            return jsonify({"reply": "감사합니다. 관심 분야, 희망 근무지, 연봉을 입력해 주세요. 예시: 품질, 서울, 3000만원"})

        if "user_text" not in state and message:
            state["user_text"] = message
            user_states[user_id] = state
            return jsonify({"reply": "감사합니다. 관심 분야, 희망 근무지, 연봉을 입력해 주세요. 예시: 품질, 서울, 3000만원"})

        # 2. 관심 분야, 지역, 연봉 입력
        if "interest" not in state and "," in message and "만원" in message:
            parts = [p.strip().replace("만원", "") for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""
            user_states[user_id] = state

        # 3. 추천 실행
        if "user_text" in state and "interest" in state:
            top_n = 1 if "더 추천해줘" in message else 3
            new_recommendations = make_recommendations(
                user_text=state["user_text"],
                interest=state.get("interest"),
                region=state.get("region"),
                salary=state.get("salary"),
                shown=state["shown"],
                top_n=top_n
            )

            if not new_recommendations:
                return jsonify({"reply": "더 이상 추천할 기업이 없습니다."})

            explanations = []
            for company, score in new_recommendations:
                reason = generate_reason_individual(state["user_text"], company, score)
                explanations.append(f"기업명: {company.get('회사명')}\n업무: {company.get('채용공고명')}\n유사도 점수: {round(score,2)}\n설명: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
            return jsonify({"reply": reply})

        return jsonify({"reply": "입력을 인식하지 못했습니다. 다시 시도해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
