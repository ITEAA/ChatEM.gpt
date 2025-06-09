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
GG_API_KEY = "8af0f404ca144249be0cfab9728b619b"
GG_API_URL = "https://openapi.gg.go.kr/EmplmntInfoStus"

user_states = {}

# Load company data
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

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

def fetch_gg_employment_info(index=1, size=100):
    print("📡 [API 호출됨] 경기도 고용정보 API 요청 중...")
    params = {"KEY": GG_API_KEY, "Type": "xml", "pIndex": index, "pSize": size}
    try:
        response = requests.get(GG_API_URL, params=params)
        root = ET.fromstring(response.content)
        rows = root.findall(".//row")

        data = []
        for row in rows:
            row_data = [row.find(col).text if row.find(col) is not None else "" for col in [
                "REGIST_DE", "SIGUN_NM", "COMPNY_NM", "EMPLMNT_TITLE", "WAGE_FORM", "SALARY_INFO",
                "WORK_REGION_LOC", "WORK_FORM", "MIN_ACDMCR", "CAREER_INFO", "CLOS_DE_INFO", "EMPLMNT_INFO_URL"
            ]]
            data.append(row_data)

        columns = ["등록일자", "시군명", "회사명", "채용공고명", "임금형태", "급여", "근무지역", "근무형태",
                   "최소학력", "경력", "마감일자", "채용정보URL"]
        df = pd.DataFrame(data, columns=columns)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"❌ 고용정보 API 오류: {e}")
        return []

def filter_companies(keywords, interest=None, region=None, salary=None):
    filtered = []
    for company in company_data:
        content = f"{company.get('name', '')} {company.get('summary', '')}".lower()
        if any(k.lower() in content for k in keywords):
            filtered.append(company)
    return filtered

def tfidf_similarity(user_text, companies):
    def get_summary(company):
        return company.get("summary") or f"{company.get('name')}에서 {company.get('summary')}"

    documents = [user_text] + [get_summary(c) for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    scored = sorted(zip(cosine_sim, companies), key=lambda x: x[0], reverse=True)
    return [(c, round(s, 2)) for s, c in scored]

def generate_reason(user_text, companies_with_scores):
    prompt = f"""
당신은 채용 컨설턴트입니다. 아래 자기소개서 내용을 바탕으로 각 기업의 직무와 왜 적합한지 분석가 입장에서 구체적으로 설명해 주세요.
각 기업마다 아래 형식에 맞춰 출력해 주세요:

기업명: OOO
업무: OOO
유사도 점수: 0.XX
설명: OOO

[자기소개서 내용]
{user_text}

[기업 목록 및 유사도 점수]
{json.dumps([{"기업명": c.get('name'), "업무": c.get('summary'), "점수": score} for c, score in companies_with_scores], ensure_ascii=False)}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content + "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!"
    except Exception as e:
        print(f"❌ GPT 설명 생성 실패: {e}")
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
            parts = [p.strip() for p in message.replace("만원", "").split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""

        if message and "user_text" not in state:
            state["user_text"] = message

        if "user_text" in state and "interest" in state:
            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, state.get("interest"), state.get("region"), state.get("salary"))
            if not filtered:
                filtered = company_data
            matched = tfidf_similarity(state["user_text"], filtered)
            selected = matched[:3]
            explanation = generate_reason(state["user_text"], selected)
            return jsonify({"reply": explanation})

        missing = []
        if "user_text" not in state:
            missing.append("자기소개서 또는 이력서")
        if "interest" not in state:
            missing.append("관심 분야, 희망 근무지, 희망 연봉")
        if missing:
            return jsonify({"reply": f"먼저 {', '.join(missing)}를 입력해 주세요."})

        return jsonify({"reply": "입력을 인식하지 못했습니다. 다시 시도해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
