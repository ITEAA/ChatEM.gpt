import os
import json
import fitz  # PyMuPDF
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"

user_states = {}

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
        keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
        return keywords
    except Exception as e:
        print(f"❌ GPT 호출 에러: {e}")
        return []

def tfidf_similarity(user_text, companies):
    if not companies:
        return []
    documents = [user_text] + [c.get("summary", "") for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    scored = sorted(zip(cosine_sim, companies), key=lambda x: x[0], reverse=True)
    return [(c, round(score, 2)) for score, c in scored if score > 0.1][:3]

def filter_companies(keywords, interest=None, region=None, salary=None):
    filtered = []
    for company in company_data:
        industry = company.get("industry", "")
        location = company.get("region", "")
        if interest and interest not in industry:
            continue
        if region and region not in location:
            continue
        filtered.append(company)
    return filtered

def generate_reason(user_text, companies_with_scores):
    companies_info = []
    for company, score in companies_with_scores:
        companies_info.append({"name": company.get("name"), "summary": company.get("summary"), "score": score})

    prompt = f"""
당신은 채용 컨설턴트 역할을 수행하고 있습니다.
아래 자기소개서와 기업 정보를 참고하여, 각 기업이 사용자에게 왜 적합한지 친절하고 전문적인 말투로 설명해 주세요.

[자기소개서 내용]
"""
{user_text}
"""

[기업 목록 및 유사도 점수]
{json.dumps(companies_info, ensure_ascii=False)}

아래 형식에 맞춰 작성해 주세요:

기업명: 설명 (자기소개서 내용 일부를 언급하며)
유사도 점수: (예: 0.85)
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ GPT 추천 설명 생성 에러: {e}")
        return "추천 이유를 생성하는 중 오류가 발생했습니다."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.remote_addr
    message = request.form.get("message", "").strip()
    interest = request.form.get("interest", "").strip()
    region = request.form.get("region", "").strip()
    salary = request.form.get("salary", "").strip()
    file = request.files.get("file")

    state = user_states.get(user_id, {})

    try:
        if file:
            user_text = extract_text_from_pdf(file)
            state = {"step": 1, "user_text": user_text}
            user_states[user_id] = state
        elif message and not state.get("user_text"):
            state = {"step": 1, "user_text": message}
            user_states[user_id] = state

        if state.get("step") == 1:
            state.update({"interest": interest, "region": region, "salary": salary})

            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, interest, region, salary)
            matched_with_scores = tfidf_similarity(state["user_text"], filtered)

            if not matched_with_scores:
                return jsonify({"reply": "조건에 맞는 기업을 찾지 못했습니다. 관심 분야나 지역을 조금 더 넓게 설정해보시겠어요?"})

            explanation = generate_reason(state["user_text"], matched_with_scores)

            user_states.pop(user_id, None)

            return jsonify({"reply": explanation})

        return jsonify({"reply": "자기소개서 또는 메시지를 먼저 입력해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
