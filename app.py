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
    # 최소 유사도를 0.7로 보정
    return [(c, round(max(score, 0.7), 2)) for score, c in scored][:3]

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
{user_text}

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
    file = request.files.get("file")
    state = user_states.get(user_id, {})

    try:
        # ✅ 1. PDF 업로드
        if file:
            user_text = extract_text_from_pdf(file)
            state["user_text"] = user_text
            user_states[user_id] = state
            return jsonify({"reply": "관심 분야, 희망 근무지, 희망 연봉을 입력해 주세요. 예시: AI, 서울, 3000만원"})

        # ✅ 2. 관심 조건 입력 추정
        if message and "," in message and "만원" in message:
            parts = [p.strip() for p in message.replace("만원", "").split(",")]
            state["interest"] = parts[0] if len(parts) > 0 else ""
            state["region"] = parts[1] if len(parts) > 1 else ""
            state["salary"] = parts[2] if len(parts) > 2 else ""
            user_states[user_id] = state

            if "user_text" in state:
                state["step"] = 2  # 모든 입력이 갖춰졌으면 추천 실행
            else:
                return jsonify({"reply": "이제 자기소개서나 이력서를 입력해 주세요."})

        # ✅ 3. 자기소개서 메시지로 입력
        if message and "user_text" not in state:
            state["user_text"] = message
            user_states[user_id] = state

            if "interest" not in state:
                return jsonify({"reply": "관심 분야, 희망 근무지, 희망 연봉을 입력해 주세요. 예시: AI, 서울, 3000만원"})
            else:
                state["step"] = 2

        if state.get("step") == 2:
            keywords = extract_keywords(state["user_text"])
            filtered = filter_companies(keywords, state.get("interest"), state.get("region"), state.get("salary"))
            if not filtered:
                print("⚠️ 조건 일치 기업 없음 → 전체 기업 중 유사도 기반 추천 진행")
                filtered = company_data
            matched_with_scores = tfidf_similarity(state["user_text"], filtered)
            if not matched_with_scores:
                return jsonify({"reply": "기업 추천에 실패했습니다. 다시 시도해 주세요."})
            explanation = generate_reason(state["user_text"], matched_with_scores)
            user_states.pop(user_id, None)
            return jsonify({"reply": explanation})

        return jsonify({"reply": "먼저 자기소개서 또는 이력서를 입력해 주세요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        return jsonify({"reply": f"❌ 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
