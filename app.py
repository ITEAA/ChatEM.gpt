import os
import json
import openai
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# OpenAI 키 설정 (환경 변수 또는 직접 입력)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 진주 기업 정보 로딩
with open("jinju_companies.json", encoding="utf-8") as f:
    JINJU_COMPANIES = json.load(f)

# 키워드 추출 함수
def extract_keywords(text):
    prompt = f"""다음 자기소개서에서 핵심 키워드를 5~7개 추출해줘. 콤마로 구분하고 형용사/명사 위주로 간결하게.
자기소개서:
{text}

키워드:"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    keywords = response.choices[0].message.content.strip()
    return [kw.strip() for kw in keywords.split(",")]

# 가장 유사한 기업 추천 함수
def recommend_companies(keywords, user_interest, region, salary):
    candidates = []
    for c in JINJU_COMPANIES:
        if region and region not in c["region"]:
            continue
        if user_interest and user_interest not in c["industry"]:
            continue
        candidates.append(c)

    if not candidates:
        return []

    corpus = [" ".join(keywords)] + [c["industry"] + " " + c["summary"] for c in candidates]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    sorted_indices = scores.argsort()[::-1]
    top_matches = [candidates[i] for i in sorted_indices[:3] if scores[i] > 0.1]
    return top_matches

# PDF 텍스트 추출 함수
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.form.get("message", "")
        interest = request.form.get("interest", "")
        region = request.form.get("region", "")
        salary = request.form.get("salary", "")
        file = request.files.get("file")

        if file:
            content = extract_text_from_pdf(file)
        else:
            content = message

        if not content.strip():
            return jsonify({"reply": "자기소개서나 메시지를 입력해 주세요."})

        keywords = extract_keywords(content)
        companies = recommend_companies(keywords, interest, region, salary)

        if not companies:
            return jsonify({"reply": "조건에 맞는 기업을 찾지 못했습니다. 입력값을 조정해 보세요."})

        reply = "아래 기업들을 추천드립니다:\n\n"
        for c in companies:
            reply += f"기업명: {c['name']}\n업종: {c['industry']}\n요약: {c['summary']}\n근무 지역: {c['region']}\n\n"
        return jsonify({"reply": reply.strip()})

    except Exception as e:
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
