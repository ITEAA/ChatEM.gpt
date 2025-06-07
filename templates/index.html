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

# 전역 상태 딕셔너리 (테스트용)
user_states = {}

# 기업 데이터 로드
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
    documents = [user_text] + [c.get("summary", "") for c in companies]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    scored = sorted(zip(cosine_sim, companies), key=lambda x: x[0], reverse=True)
    return [c for score, c in scored if score > 0.1][:3]

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

def generate_reason(user_text, companies):
    prompt = f"""
    다음 자기소개서를 참고해서 아래의 기업들을 추천하는 이유를 간단히 설명해줘.

    자기소개서:
    {user_text}

    기업 목록:
    {json.dumps(companies, ensure_ascii=False)}

    결과는 각 기업에 대해 \"기업명: 추천 사유\" 형식으로 출력해줘.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
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
    user_id = request.remote_addr  # 사용자 구분 ID (간단 버전)
    message = request.form.get("message", "").strip()
    interest = request.form.get("interest", "").strip()
    region = request.form.get("region", "").strip()
    salary = request.form.get("salary", "").strip()
    file = request.files.get("file")

    state = user_states.get(user_id, {})

    try:
        # Step 0: 파일 업로드 or 메시지 입력
        if file:
            user_text = extract_text_from_pdf(file)
            state = {"step": 1, "user_text": user_text}
            user_states[user_id] = state
        elif message and not state.get("user_text"):
            state = {"step": 1, "user_text": message}
            user_states[user_id] = state

        # Step 1: 정보 수집 (관심 분야, 지역)
        if state.get("step") == 1:
            state.update({"interest": interest, "region": region, "salary": salary})

            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, interest, region, salary)
            matched = tfidf_similarity(state["user_text"], filtered)
            explanation = generate_reason(state["user_text"], matched)

            user_states.pop(user_id, None)

            return jsonify({"reply": explanation})

        return jsonify({"reply": "자기소개서 또는 메시지를 먼저 입력해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
