from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import fitz  # PyMuPDF
from openai import OpenAI
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
client = OpenAI(api_key=api_key)

# 기업 데이터 로딩
with open("jinju_companies.json", "r", encoding="utf-8") as f:
    company_data = json.load(f)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text.strip()

def extract_keywords(text):
    prompt = f"""
    다음 자기소개서 또는 이력서에서 핵심 키워드를 추출해줘.
    - 5~10개 정도 뽑아줘.
    - 키워드는 컴마(,)로 구분해서 출력해줘.

    내용:
    {text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        result = response.choices[0].message.content
        keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
        return keywords
    except Exception as e:
        print(f"❌ GPT 호출 에러: {e}")
        return []

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def match_companies(keywords, interest=None, region=None, salary=None):
    matches = []

    for company in company_data:
        industry = company.get("industry", "")
        summary = company.get("summary", "")
        location = company.get("region", "")

        score = 0

        # 단어 유사도 기반 매칭 점수
        for kw in keywords:
            if similarity(kw, industry) > 0.7:
                score += 2
            elif similarity(kw, summary) > 0.5:
                score += 1

        # 관심 산업/지역 기반 보너스
        if interest and similarity(interest, industry) > 0.7:
            score += 1
        if region and similarity(region, location) > 0.7:
            score += 1

        if score > 0:
            matches.append((score, company))

    sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)
    top_companies = [c for _, c in sorted_matches[:3]]
    return top_companies

def generate_response(keywords, companies):
    if not companies:
        return "조건에 맞는 회사를 찾기 어려웠습니다. 다른 입력값으로 다시 시도해보세요."

    response_lines = ["다음은 추천 기업입니다:"]
    for c in companies:
        line = f"\n\n📌 기업명: {c['name']}\n산업 분야: {c['industry']}\n근무 지역: {c['region']}"
        if c.get("summary"):
            line += f"\n주요 내용: {c['summary']}"
        if c.get("url"):
            line += f"\n채용공고: {c['url']}"
        response_lines.append(line)
    return "\n".join(response_lines)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    interest = request.form.get("interest", "").strip()
    region = request.form.get("region", "").strip()
    salary = request.form.get("salary", "").strip()
    file = request.files.get("file")

    try:
        if file:
            text = extract_text_from_pdf(file)
        else:
            text = message

        if not text:
            return jsonify({"reply": "자기소개서나 메시지를 입력해 주세요."})

        keywords = extract_keywords(text)
        companies = match_companies(keywords, interest, region, salary)
        reply = generate_response(keywords, companies)
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
