import os
import json
import fitz  # PyMuPDF
import openai
import uuid
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)
app.secret_key = "change-this-to-a-super-secret-random-string"

# API 키 로딩
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
openai.api_key = api_key

GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {}

try:
    with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
        cached_companies = json.load(f)
except Exception:
    cached_companies = []

# KoBERT 모델 로딩 (trust_remote_code=True 추가)
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    print(f"KoBERT 모델 로딩 실패: {e}")
    raise e

# 기업 임베딩 미리 계산
def get_kobert_embedding_for_startup(text):
    if not text:
        return torch.zeros(model.config.hidden_size).to(device)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    except Exception as e:
        print(f"임베딩 오류: {e}")
        return torch.zeros(model.config.hidden_size).to(device)

for company in cached_companies:
    summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
    company['embedding'] = get_kobert_embedding_for_startup(summary)

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
company_summaries_for_tfidf = [
    f"{company.get('채용공고명', '')} {company.get('회사명', '')}" for company in cached_companies
]
company_tfidf_matrix = tfidf_vectorizer.fit_transform(company_summaries_for_tfidf)

# PDF 텍스트 추출
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()

# GPT 키워드 추출
def extract_keywords(text):
    prompt = f"다음 자기소개서 또는 이력서에서 5~10개의 핵심 키워드를 콤마(,)로 추출:\n{text}"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return [kw.strip() for kw in response.choices[0].message.content.split(",") if kw.strip()]
    except Exception:
        return []

# 사용자용 KoBERT 임베딩
def get_kobert_embedding(text):
    return get_kobert_embedding_for_startup(text)

# 유사도 계산
def kobert_similarity(user_text, companies):
    user_embedding = get_kobert_embedding(user_text).cpu().numpy().reshape(1, -1)
    results = []
    for c in companies:
        company_embedding = c.get('embedding')
        if company_embedding is not None and not torch.all(company_embedding == 0):
            score = cosine_similarity(user_embedding, company_embedding.cpu().numpy().reshape(1, -1))[0][0]
            results.append((c, float(score)))
    return sorted(results, key=lambda x: x[1], reverse=True)

def tfidf_similarity(user_text, companies):
    user_vector = tfidf_vectorizer.transform([user_text])
    scores = cosine_similarity(user_vector, company_tfidf_matrix).flatten()
    return [(companies[i], float(scores[i])) for i in range(len(scores))]

# GPT 설명 생성
def generate_reason_individual(user_text, company, score):
    prompt = f"""
[사용자 자기소개서]
{user_text}

[기업 정보]
- 기업명: {company.get('회사명')}
- 채용공고: {company.get('채용공고명')}
- 유사도 점수: {round(score, 2)}

[설명]
사용자의 자기소개서를 기반으로 이 기업을 추천하는 이유를 설명해줘.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "설명 생성 실패"

# Hybrid 추천 로직
def make_recommendations(user_text, interest, region, salary, shown=set(), top_n=3):
    kobert_scores = kobert_similarity(user_text, cached_companies)
    tfidf_scores = tfidf_similarity(user_text, cached_companies)
    tfidf_dict = {(c['회사명'], c['채용공고명']): score for c, score in tfidf_scores}

    results = []
    for company, k_score in kobert_scores:
        key = (company['회사명'], company['채용공고명'])
        t_score = tfidf_dict.get(key, 0)
        final = 0.7 * k_score + 0.3 * t_score

        # 가산점 필터 (관심분야, 지역 등)
        passes = True
        summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        if interest and interest not in summary:
            passes = False
        if region and region not in company.get("근무지역", ""):
            passes = False

        if passes and key not in shown:
            shown.add(key)
            results.append((company, final))
        if len(results) >= top_n:
            break
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    message = request.form.get("message", "").strip()
    file = request.files.get("file")
    state = user_states.get(user_id, {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None})

    try:
        if file:
            state["user_text"] = extract_text_from_pdf(file)
            user_states[user_id] = state
            return jsonify({"reply": "이력서 분석 완료. 관심 분야, 지역, 연봉을 입력해 주세요. (예: 품질, 서울, 3000만원)"})

        if state["user_text"] is None and message:
            if len(message.split()) > 30 or "이력서" in message or "자기소개서" in message:
                state["user_text"] = message
                user_states[user_id] = state
                return jsonify({"reply": "이력서 분석 완료. 관심 분야, 지역, 연봉을 입력해 주세요. (예: 품질, 서울, 3000만원)"})
            else:
                return jsonify({"reply": "자기소개서 또는 이력서 파일이나 내용을 입력해 주세요."})

        if state["user_text"] and state["interest"] is None and "," in message:
            parts = [p.strip() for p in message.split(",")]
            state["interest"] = parts[0] if parts[0] != "없음" else ""
            state["region"] = parts[1] if len(parts) > 1 and parts[1] != "없음" else ""
            state["salary"] = parts[2].replace("만원", "") if len(parts) > 2 and parts[2] != "없음" else ""
            user_states[user_id] = state

            results = make_recommendations(state["user_text"], state["interest"], state["region"], state["salary"], state["shown"], 3)
            if not results:
                return jsonify({"reply": "조건에 맞는 기업이 없습니다. 다른 조건을 입력해 주세요."})

            reply_parts = []
            for company, score in results:
                c_info = {k: v for k, v in company.items() if k != 'embedding'}
                reason = generate_reason_individual(state["user_text"], c_info, score)
                reply_parts.append(f"**기업명**: {c_info.get('회사명')}\n**채용공고**: {c_info.get('채용공고명')}\n**점수**: {round(score,2)}\n**설명**: {reason}\n")
            return jsonify({"reply": "\n\n".join(reply_parts)})

        if "더 추천해줘" in message:
            results = make_recommendations(state["user_text"], state["interest"], state["region"], state["salary"], state["shown"], 1)
            if not results:
                return jsonify({"reply": "더 추천할 기업이 없습니다."})
            company, score = results[0]
            c_info = {k: v for k, v in company.items() if k != 'embedding'}
            reason = generate_reason_individual(state["user_text"], c_info, score)
            return jsonify({"reply": f"**기업명**: {c_info.get('회사명')}\n**채용공고**: {c_info.get('채용공고명')}\n**점수**: {round(score,2)}\n**설명**: {reason}"})

        if "추천 초기화" in message:
            user_states[user_id] = {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None}
            return jsonify({"reply": "추천 상태가 초기화되었습니다. 새로운 파일을 첨부해 주세요."})

        return jsonify({"reply": "무슨 말씀인지 잘 이해하지 못했어요. 다시 입력해 주세요."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"reply": f"오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
