from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import fitz  # PyMuPDF
import openai
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# OpenAI API 키 설정 (환경 변수 또는 직접 설정)
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"

# 기업 데이터 로딩 (진주 지역)
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
        response = openai.ChatCompletion.create(
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

def match_companies(keywords, interest=None, region=None, salary=None):
    matches = []

    for company in company_data:
        industry = company.get("industry", "")
        location = company.get("region", "")

        score = 0
        # 키워드 매칭 점수
        for kw in keywords:
            if kw in industry:
                score += 2
            elif kw in company.get("summary", ""):
                score += 1

        # 관심 산업/지역 기반 보너스 점수
        if interest and interest in industry:
            score += 1
        if region and region in location:
            score += 1

        if score > 0:
            matches.append((score, company))

    # 점수 기준 내림차순 정렬 후 상위 3개 추천
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
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
