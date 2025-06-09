import os
import json
import fitz # PyMuPDF
import openai
import uuid
import re # 정규표현식 모듈 추가
import traceback # 오류 스택 트레이스 출력을 위한 모듈 추가

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-super-secret-random-key-here-for-production")

# --- API 키 로딩 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 프로그램을 시작할 수 없습니다.")
openai.api_key = api_key

GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {} # 사용자별 대화 상태를 저장할 딕셔너리 (인메모리)

# KoBERT 모델 및 장치 변수 초기화
tokenizer = None
model = None
device = None

# 기업 정보 및 벡터 초기화
cached_companies = []
tfidf_vectorizer = None
company_tfidf_matrix = None

# --- KoBERT 임베딩 생성 함수 (단일 함수로 통합) ---
def get_kobert_embedding(text_input):
    if model is None or tokenizer is None or device is None or not text_input:
        return torch.zeros(768).to(device) # KoBERT 기본 hidden_size
    try:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        return embedding
    except Exception as e:
        print(f"❌ 임베딩 생성 오류 (텍스트: '{text_input[:30]}...'): {e}")
        traceback.print_exc()
        return torch.zeros(768).to(device)

try:
    if not os.path.exists(GG_CACHED_FILE):
        print(f"경고: '{GG_CACHED_FILE}' 파일을 찾을 수 없습니다. 빈 회사 목록으로 시작합니다.")
    else:
        with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
            cached_companies = json.load(f)
        print(f"✅ '{GG_CACHED_FILE}'에서 {len(cached_companies)}개 기업 정보 로드 성공.")

    # KoBERT 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("✅ KoBERT 모델 로드 성공!")

    # --- 서버 시작 시 모든 기업 정보에 대한 KoBERT 임베딩 미리 계산 ---
    print("서버 시작 전, 기업 정보 KoBERT 임베딩을 시작합니다... (데이터 양에 따라 몇 분 소요될 수 있습니다)")
    for company in cached_companies:
        summary_text = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        company['embedding'] = get_kobert_embedding(summary_text) # 통합된 함수 사용
    print("✅ 모든 기업 정보의 KoBERT 임베딩이 완료되어 메모리에 저장되었습니다.")

    # --- TF-IDF Vectorizer 및 기업별 TF-IDF 행렬 미리 계산 ---
    print("서버 시작 전, 기업 정보 TF-IDF 벡터화를 시작합니다...")
    tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    company_summaries_for_tfidf = [
        f"{company.get('채용공고명', '')} {company.get('회사명', '')}" for company in cached_companies
    ]
    company_tfidf_matrix = tfidf_vectorizer.fit_transform(company_summaries_for_tfidf)
    print("✅ 모든 기업 정보의 TF-IDF 벡터화가 완료되었습니다.")

except json.JSONDecodeError as e:
    print(f"❌ 오류: '{GG_CACHED_FILE}' 파일이 유효한 JSON 형식이 아닙니다. 오류: {e}")
    cached_companies = []
    raise RuntimeError(f"기업 정보 파일 로딩에 실패했습니다. 오류: {e}")
except Exception as e:
    print(f"❌ 초기 설정 중 치명적인 오류 발생: {e}")
    traceback.print_exc()
    raise RuntimeError(f"애플리케이션 초기 설정에 실패했습니다. 오류: {e}")


# --- PDF에서 텍스트 추출 함수 ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        doc = fitz.open(stream=pdf_file_stream.read(), filetype="pdf")
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        raw_text = "\n".join(text_content)

        processed_text = re.sub(r'\s+', ' ', raw_text).strip()

        # 디버깅 출력은 그대로 유지
        print("\n--- PDF에서 추출된 원본 텍스트 (앞부분 500자) ---")
        print(raw_text[:500])
        print("-------------------------------------------\n")
        print("--- PDF에서 추출 및 전처리된 텍스트 (앞부분 500자) ---")
        print(processed_text[:500])
        print("---------------------------------------------------\n")

        return processed_text
    except Exception as e:
        print(f"❌ PDF 텍스트 추출 오류: {e}")
        traceback.print_exc()
        return ""


# --- GPT를 사용하여 키워드 추출 함수 ---
def extract_keywords(text):
    if not text:
        return []

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
        traceback.print_exc()
        return []


# --- KoBERT 유사도 계산 함수 ---
def kobert_similarity(user_text, companies):
    if not user_text or not companies:
        return []
    user_embedding = get_kobert_embedding(user_text)
    user_embedding_np = user_embedding.cpu().numpy().reshape(1, -1)

    results = []
    for c in companies:
        company_embedding = c.get('embedding')
        if company_embedding is not None and not torch.all(company_embedding == 0):
            company_embedding_np = company_embedding.cpu().numpy().reshape(1, -1)
            score = cosine_similarity(user_embedding_np, company_embedding_np)[0][0]
            results.append((c, float(score)))
    return sorted(results, key=lambda x: x[1], reverse=True)

# --- TF-IDF 유사도 계산 함수 ---
def tfidf_similarity(user_text, companies):
    if not user_text or not companies or tfidf_vectorizer is None or company_tfidf_matrix is None:
        return []
    try:
        user_tfidf_vector = tfidf_vectorizer.transform([user_text])
        scores = cosine_similarity(user_tfidf_vector, company_tfidf_matrix).flatten()
        results = [(companies[i], float(scores[i])) for i in range(len(scores))]
        return results
    except Exception as e:
        print(f"❌ TF-IDF 유사도 계산 오류: {e}")
        traceback.print_exc()
        return []

# --- GPT를 사용하여 추천 이유 생성 함수 ---
def generate_reason_individual(user_text, company, score):
    prompt = f"""
    당신은 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 매칭시켜주는 전문가입니다.
    사용자의 자기소개서 내용과 기업 정보를 기반으로, 왜 이 기업이 사용자에게 적합한지 상세하게 설명해주세요.
    친절하고 전문적인 어조로, 설명은 자연스럽고 명확하게 전달해야 합니다.
    시스템 작동 관련 문구("분석모드입니다" 등)는 절대 사용하지 마세요.
    분석이 어려운 경우 "현재 정보만으로는 분석이 어렵습니다"와 같이 자연스럽게 안내해주세요.

    [기업 정보]
    - 기업명: {company.get('회사명', '정보 없음')}
    - 채용공고명: {company.get('채용공고명', '정보 없음')}
    - 유사도 점수: {round(score, 2)}

    [사용자 자기소개서]
    {user_text}

    [설명 시작]
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT 설명 생성 오류: {e}")
        traceback.print_exc()
        return "설명을 생성하는 데 문제가 발생했습니다."

# --- 연봉 정보 파싱 헬퍼 함수 ---
def parse_salary_info(summary_text):
    min_salary = 0
    max_salary = float('inf')

    match_annual = re.search(r'연봉 (\d+)(?:만원)?(?: ~ (\d+)(?:만원)?)?', summary_text)
    if match_annual:
        min_salary = int(match_annual.group(1))
        max_salary = int(match_annual.group(2)) if match_annual.group(2) else min_salary
        return min_salary, max_salary

    match_monthly = re.search(r'월급 (\d+)(?:만원)?(?: ~ (\d+)(?:만원)?)?', summary_text)
    if match_monthly:
        min_monthly = int(match_monthly.group(1))
        min_salary = min_monthly * 12
        max_monthly = int(match_monthly.group(2)) if match_monthly.group(2) else min_monthly
        max_salary = max_monthly * 12
        return min_salary, max_salary

    match_hourly = re.search(r'시급 (\d+)', summary_text)
    if match_hourly:
        hourly_wage = int(match_hourly.group(1))
        min_salary = (hourly_wage * 209 * 12) / 10000
        max_salary = min_salary
        return int(min_salary), int(max_salary)

    return 0, float('inf')

# --- 기업 필터링 로직 함수 ---
def apply_company_filters(company, interest, region, salary):
    passes_filter = True

    if interest:
        summary_lower = company.get('summary', '').lower()
        industry_lower = company.get('industry', '').lower()
        interest_lower = interest.lower()
        if interest_lower not in summary_lower and interest_lower not in industry_lower:
            passes_filter = False

    if region and region.lower() not in company.get("region", "").lower():
        passes_filter = False

    if salary:
        try:
            min_salary_req = int(salary)
            company_min_salary, company_max_salary = parse_salary_info(company.get("summary", ""))
            if min_salary_req > company_max_salary:
                passes_filter = False
        except ValueError:
            pass # 유효하지 않은 연봉 입력 또는 파싱 오류는 무시

    return passes_filter

# --- 기업 추천 로직 함수 (Hybrid 모델) ---
def make_recommendations(user_text, interest, region, salary, shown_companies_set=None, top_n=3):
    if shown_companies_set is None:
        shown_companies_set = set()

    if not user_text or not cached_companies:
        return []

    kobert_ranked_companies = kobert_similarity(user_text, cached_companies)
    tfidf_ranked_companies = tfidf_similarity(user_text, cached_companies)

    tfidf_scores = {
        (c.get("name"), c.get("summary")): score
        for c, score in tfidf_ranked_companies
    }

    hybrid_scores = []
    for company, kobert_score in kobert_ranked_companies:
        company_key = (company.get("name"), company.get("summary"))
        tfidf_score = tfidf_scores.get(company_key, 0.0)

        kobert_weight = 0.7
        tfidf_weight = 0.3
        final_score = (kobert_weight * kobert_score) + (tfidf_weight * tfidf_score)

        if apply_company_filters(company, interest, region, salary):
            hybrid_scores.append((company, final_score))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for comp, sim in hybrid_scores:
        comp_id_str = json.dumps((comp.get("name"), comp.get("summary")), ensure_ascii=False)
        if comp_id_str not in shown_companies_set:
            shown_companies_set.add(comp_id_str)
            results.append((comp, sim))
        if len(results) >= top_n:
            break

    return results

# --- 사용자 상태 관리 헬퍼 함수 ---
def _get_user_state(user_id):
    return user_states.get(user_id, {
        "shown": set(),
        "user_text": None,
        "interest": None,
        "region": None,
        "salary": None
    })

def _update_user_state(user_id, state):
    user_states[user_id] = state

# --- 추천 결과 응답 생성 헬퍼 함수 (중복 로직 제거) ---
def _generate_recommendation_response(user_text, recommendations, additional_message=""):
    if not recommendations:
        return "아쉽게도 현재 조건에 맞는 기업을 찾을 수 없습니다. 다른 조건을 말씀해주시거나 '추천 초기화'를 통해 다시 시작해 주시겠어요?"

    explanations = []
    for company, score in recommendations:
        company_info_for_gpt = {k: v for k, v in company.items() if k not in ['embedding', 'summary']}
        reason = generate_reason_individual(user_text, company_info_for_gpt, score)
        explanations.append(f"**기업명**: {company_info_for_gpt.get('name', '정보 없음')}\n**채용공고명**: {company_info_for_gpt.get('summary', '정보 없음')}\n**종합 점수**: {round(score,2)}\n**설명**: {reason}\n")
    
    reply = "\n\n".join(explanations)
    reply += f"\n\n📌 {additional_message if additional_message else '더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요!'}"
    return reply

# --- Chat 라우트 핸들러 헬퍼 함수들 ---
def _handle_pdf_upload(file_stream, user_id, state):
    user_text = extract_text_from_pdf(file_stream)
    if user_text:
        state["user_text"] = user_text
        state["shown"] = set() # 파일 업로드 시 초기화
        _update_user_state(user_id, state)
        return {"reply": "감사합니다. 이력서/자기소개서 내용이 성공적으로 분석되었습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 입력해 주세요."}
    else:
        return {"reply": "PDF 파일에서 텍스트를 추출하는 데 실패했습니다. 파일이 스캔된 이미지 기반이거나 손상되었을 수 있습니다. 텍스트를 직접 입력해 주시거나 다른 파일을 시도해 주시겠어요?"}

def _handle_initial_text_input(message, user_id, state):
    state["user_text"] = message
    state["shown"] = set() # 텍스트 직접 입력 시 초기화
    _update_user_state(user_id, state)
    return {"reply": "이력서/자기소개서 내용을 확인했습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 입력해 주세요."}

def _handle_preference_input(message, user_id, state):
    if "," in message:
        parts = [p.strip() for p in message.split(",")]
        state["interest"] = parts[0] if len(parts) > 0 and parts[0].lower() != "없음" else ""
        state["region"] = parts[1] if len(parts) > 1 and parts[1].lower() != "없음" else ""
        state["salary"] = parts[2].replace("만원", "") if len(parts) > 2 and parts[2].lower() != "없음" else ""
        _update_user_state(user_id, state)

        new_recommendations = make_recommendations(
            user_text=state["user_text"],
            interest=state.get("interest"),
            region=state.get("region"),
            salary=state.get("salary"),
            shown_companies_set=state["shown"],
            top_n=3
        )
        return {"reply": _generate_recommendation_response(
            state["user_text"],
            new_recommendations,
            "더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 예를 들어 '더 추천해줘'라고 말씀하시면 다른 기업을 찾아드릴 수 있습니다."
        )}
    else:
        return {"reply": "관심 분야, 희망 근무지, 연봉을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 콤마(,)로 구분해서 입력해 주세요."}

def _handle_more_recommendations(user_id, state):
    new_recommendations = make_recommendations(
        user_text=state["user_text"],
        interest=state.get("interest"),
        region=state.get("region"),
        salary=state.get("salary"),
        shown_companies_set=state["shown"],
        top_n=1
    )
    return {"reply": _generate_recommendation_response(
        state["user_text"],
        new_recommendations,
        "더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 또는 '추천 초기화'라고 말씀하시면 처음부터 다시 시작할 수 있습니다."
    )}

def _handle_reset(user_id):
    user_states.pop(user_id, None)
    return {"reply": "추천 상태가 초기화되었습니다. 새로운 자기소개서/이력서 파일을 첨부하시거나 내용을 직접 입력해 주세요."}

# --- Flask 라우트 설정 ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    state = _get_user_state(user_id)

    message = request.form.get("message", "").strip()
    file = request.files.get("file")

    try:
        # 1. 파일 첨부 처리
        if file and file.filename != '':
            return jsonify(_handle_pdf_upload(file, user_id, state))

        # 2. 자기소개서/이력서 텍스트 입력 처리 (아직 자기소개서가 없는 경우)
        if state["user_text"] is None:
            if len(message.split()) > 30 or "이력서" in message or "자기소개서" in message:
                return jsonify(_handle_initial_text_input(message, user_id, state))
            else:
                return jsonify({"reply": "개인별 맞춤 분석을 위해서는 자기소개서 혹은 이력서가 필요합니다. 파일을 첨부해 주시거나 내용을 직접 입력해 주시면 상세한 분석을 제공해 드리겠습니다."})

        # 3. 사용자 선호도(관심 분야, 지역, 연봉) 입력 처리 (자기소개서가 있고 선호도가 없는 경우)
        if state["interest"] is None:
            return jsonify(_handle_preference_input(message, user_id, state))

        # 4. "더 추천해줘" 요청 처리
        if "더 추천
