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

# Load cached job data
with open("gg_employment_cached.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

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
        base = 0
        summary = company.get("summary", "")
        region_val = company.get("근무지역", "")
        salary_val = company.get("급여", "")
        if interest and interest in summary:
            base += 0.3
        if region and region in region_val:
            base += 0.3
        if salary and salary in salary_val:
            base += 0.2
        return base
    return sorted(company_data, key=score, reverse=True)

def tfidf_similarity(user_text, companies):
    documents = [user_text] + [c.get("summary") or f"{c.get('회사명')} {c.get('채용공고명')}" for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return sorted(zip(companies, cosine_sim), key=lambda x: x[1], reverse=True)

def generate_reason(user_text, companies_with_scores):
    companies_info = [
        {
            "name": c.get("회사명"),
            "summary": c.get("summary") or c.get("채용공고명"),
            "score": round(score, 2),
            "url": c.get("채용정보URL")
        }
        for c, score in companies_with_scores
    ]

    prompt = f"""
당신은 채용 컨설턴트입니다.
다음 자기소개서 내용을 바탕으로, 각 기업이 사용자의 경력과 얼마나 잘 맞는지 설명해 주세요.
다음 형식으로 출력해주세요:

기업명: OOO
업무: OOO
유사도 점수: 0.XX
설명: (분석가의 시선에서 자기소개서의 특정 경험과 직무 연결)

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
        return response.choices[0].message.content + "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
    except Exception as e:
        print(f"❌ GPT 설명 생성 오류: {e}")
        return "추천 설명 생성 중 오류가 발생했습니다."

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

        # 모든 정보 수집 완료 후 추천
        if all(k in state for k in ("user_text", "interest", "region", "salary")) and "matched" not in state:
            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, state["interest"], state["region"], state["salary"])
            matched = tfidf_similarity(state["user_text"], filtered)
            state["matched"] = matched
            state["recommended"] = set()
            top3 = matched[:3]
            state["recommended"].update([c[0]["회사명"] for c in top3])
            explanation = generate_reason(state["user_text"], top3)
            user_states[user_id] = state
            return jsonify({"reply": explanation})

        # "더 추천해줘" 요청 처리
        if "더 추천해줘" in message and "matched" in state:
            remaining = [(c, s) for c, s in state["matched"] if c["회사명"] not in state["recommended"]]
            if remaining:
                next_one = remaining[0]
                state["recommended"].add(next_one[0]["회사명"])
                explanation = generate_reason(state["user_text"], [next_one])
                user_states[user_id] = state
                return jsonify({"reply": explanation})
            else:
                return jsonify({"reply": "✅ 더 이상 추천할 기업이 없습니다."})

        # 조건이 아직 부족한 경우
        missing = []
        if "user_text" not in state:
            missing.append("자기소개서 또는 이력서")
        if not all(k in state for k in ("interest", "region", "salary")):
            missing.append("관심 분야, 희망 근무지, 희망 연봉")
        if missing:
            return jsonify({"reply": f"먼저 {', '.join(missing)}를 입력해 주세요."})

        return jsonify({"reply": "입력을 인식하지 못했습니다. 다시 시도해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
