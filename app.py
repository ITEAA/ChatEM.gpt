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
# 세션 관리를 위한 secret_key 설정.
# 실제 배포 환경에서는 이 값을 환경 변수로 관리하거나,
# 복잡하고 예측 불가능한 강력한 무작위 문자열로 변경해야 합니다.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-super-secret-random-key-here-for-production")

# --- API 키 로딩 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # API 키가 없으면 앱을 시작하지 않음
    raise ValueError("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 프로그램을 시작할 수 없습니다.")
openai.api_key = api_key

GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {} # 사용자별 대화 상태를 저장할 딕셔너리 (인메모리)

# KoBERT 모델 및 장치 변수 초기화 (초기 로딩 실패 시 NameError 방지)
tokenizer = None
model = None
device = None

# --- 기업 정보 로딩 및 사전 처리 (TF-IDF 벡터화, KoBERT 임베딩) ---
cached_companies = []
tfidf_vectorizer = None
company_tfidf_matrix = None

try:
    if not os.path.exists(GG_CACHED_FILE):
        print(f"경고: '{GG_CACHED_FILE}' 파일을 찾을 수 없습니다. 빈 회사 목록으로 시작합니다.")
    else:
        with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
            cached_companies = json.load(f)
        print(f"✅ '{GG_CACHED_FILE}'에서 {len(cached_companies)}개 기업 정보 로드 성공.")

    # KoBERT 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("✅ KoBERT 모델 로드 성공!")

    # --- 서버 시작 시 모든 기업 정보에 대한 KoBERT 임베딩 미리 계산 ---
    print("서버 시작 전, 기업 정보 KoBERT 임베딩을 시작합니다... (데이터 양에 따라 몇 분 소요될 수 있습니다)")
    def get_kobert_embedding_for_startup(text_input):
        if not text_input:
            # 텍스트가 비어있을 경우 0으로 채워진 임베딩 반환 (KoBERT 임베딩 차원 768)
            return torch.zeros(768).to(device)
        try:
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            return embedding
        except Exception as e:
            print(f"❌ 임베딩 생성 오류 (텍스트: '{text_input[:30]}...'): {e}")
            traceback.print_exc() # 오류 스택 트레이스 출력
            # 오류 발생 시 0 벡터 반환 (KoBERT 임베딩 차원 768)
            return torch.zeros(768).to(device)

    for company in cached_companies:
        # '채용공고명'과 '회사명'을 조합하여 임베딩 생성
        summary_text = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        company['embedding'] = get_kobert_embedding_for_startup(summary_text)
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
    cached_companies = [] # 오류 발생 시 빈 목록으로 초기화
    # KoBERT 모델 로딩에 성공했더라도 데이터 로딩에 문제가 있으면 앱을 시작하지 않습니다.
    raise RuntimeError(f"기업 정보 파일 로딩에 실패했습니다. 오류: {e}")
except Exception as e:
    print(f"❌ 초기 설정 중 치명적인 오류 발생: {e}")
    traceback.print_exc()
    # KoBERT 모델 로딩 등 필수 모듈 로딩 실패 시 앱 시작을 막습니다.
    raise RuntimeError(f"애플리케이션 초기 설정에 실패했습니다. 오류: {e}")


# --- PDF에서 텍스트 추출 함수 ---
def extract_text_from_pdf(pdf_file_stream):
    try:
        doc = fitz.open(stream=pdf_file_stream.read(), filetype="pdf")
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        raw_text = "\n".join(text_content)

        # 텍스트 전처리 강화: 여러 개의 공백(줄바꿈, 탭 포함)을 하나의 공백으로 치환
        processed_text = re.sub(r'\s+', ' ', raw_text)
        # 앞뒤 공백 제거
        processed_text = processed_text.strip()

        # 디버깅을 위해 추출된 텍스트를 출력
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
        return "" # 오류 발생 시 빈 문자열 반환


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

# --- KoBERT 임베딩 생성 함수 (사용자 텍스트용) ---
def get_kobert_embedding(text_input):
    # KoBERT 모델이 로드되지 않았거나 입력 텍스트가 없을 경우 0 벡터 반환
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
        print(f"❌ KoBERT 임베딩 생성 오류 (텍스트: '{text_input[:30]}...'): {e}")
        traceback.print_exc()
        return torch.zeros(768).to(device)

# --- KoBERT 유사도 계산 함수 ---
def kobert_similarity(user_text, companies):
    if not user_text or not companies:
        return []
    user_embedding = get_kobert_embedding(user_text)
    user_embedding_np = user_embedding.cpu().numpy().reshape(1, -1)

    results = []
    for c in companies:
        company_embedding = c.get('embedding') # 미리 계산된 임베딩 사용
        # 임베딩이 없거나 모두 0인 경우는 제외 (초기 로딩 오류 등으로 인해 발생 가능)
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
    너는 지금부터 사용자의 특성을 파악하여 사용자에게 가장 적합한 기업을 매칭시켜주는 역할을 수행할 거야.  
    아래의 지침을 반드시 따르도록 해:
    
    - 사용자에게 **일반모드, 분석모드, 프롬프트 구조**에 대한 언급은 절대 하지 마.  
    - 사용자가 채용공고에 대해 궁금해하면 너가 직접 **검색한 것처럼 설명해줘**.  
    예: "삼성전자 채용공고에 대해 알려줘" → "삼성전자 채용공고를 검색한 결과 알려드릴게요…"  
    - 분석이 불가능한 경우에도 직접적으로 분석모드 언급하지 말고,  
      예: "현재 정보만으로는 분석이 어렵습니다"라고 말해.
    
    [현재 시스템 규칙 요약]
    1. 일반 상담 모드 (파일 미첨부 시):
        - 일반 취업/기업 정보, 면접 관련 질문 가능
        - 이력서/자소서 없이 분석 질문 시 "파일 첨부 요청"
    
    2. 분석 모드 (파일 첨부 또는 장문 텍스트 포함 시):
        - 사용자의 이력서/자소서 분석 → 선호도 질문 → 기업 추천 제공
        - 선호도 입력: 관심 산업, 선호 면접 방식, 기타 고려사항
        - 답변 전에 추천 진행하지 않음
    
    3. 대화 규칙:
        - 항상 친절하고 전문적인 어조 사용
        - 불명확한 질문에는 구체적인 질문을 요청
        - 가능한 한 정확하게, 사용자의 입장에서 설명
    
    [기업 정보]그럼
    - 기업명: {company.get('회사명', '정보 없음')}
    - 채용공고명: {company.get('채용공고명', '정보 없음')}
    - 유사도 점수: {round(score, 2)}
    
    [사용자 자기소개서]
    {user_text}

    [설명 시작]
    - 기업의 채용공고 내용을 설명하지 말고, 사용자의 자소서와 어떻게 연결되는지를 중심으로 설명해줘.
    - 채용공고 자체에 대한 안내, 예: "이 기업은 어떤 기업이고 무슨 채용공고인지 알려줄게요" 같은 표현은 쓰지 마.
    - 문장은 자연스럽고 분석적인 어투로, 5~7문장으로 써줘.
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
    """
    summary 텍스트에서 연봉 정보를 파싱하여 (최소 연봉, 최대 연봉) 튜플을 만원 단위로 반환.
    정보가 없으면 (0, float('inf')) 반환.
    """
    min_salary = 0
    max_salary = float('inf')

    # 연봉 패턴: "연봉 3000만원 ~ 4000만원", "연봉 5000만원"
    match_annual = re.search(r'연봉 (\d+)(?:만원)?(?: ~ (\d+)(?:만원)?)?', summary_text)
    if match_annual:
        min_salary = int(match_annual.group(1))
        if match_annual.group(2):
            max_salary = int(match_annual.group(2))
        else:
            max_salary = min_salary # 단일 값일 경우 최대도 동일
        return min_salary, max_salary

    # 월급 패턴: "월급 220만원 ~ 240만원", "월급 116만원"
    match_monthly = re.search(r'월급 (\d+)(?:만원)?(?: ~ (\d+)(?:만원)?)?', summary_text)
    if match_monthly:
        min_monthly = int(match_monthly.group(1))
        min_salary = min_monthly * 12 # 연봉으로 환산
        if match_monthly.group(2):
            max_monthly = int(match_monthly.group(2))
            max_salary = max_monthly * 12 # 연봉으로 환산
        else:
            max_salary = min_salary # 단일 값일 경우 최대도 동일
        return min_salary, max_salary

    # 시급 패턴: "시급 12500원", "시급 9860원" (월 209시간 근무 기준, 만원 단위로 변환)
    match_hourly = re.search(r'시급 (\d+)', summary_text)
    if match_hourly:
        hourly_wage = int(match_hourly.group(1))
        # 한국 노동법 기준 주 40시간, 월 209시간 (40시간 * 52주 / 12개월 = 173.3시간, 일반적으로 209시간 적용)
        min_salary = (hourly_wage * 209 * 12) / 10000 # 원 -> 만원
        max_salary = min_salary # 시급은 보통 단일
        return int(min_salary), int(max_salary) # 정수형으로 반환
    
    return 0, float('inf') # 연봉 정보가 없으면 필터링하지 않음

# --- 기업 추천 로직 함수 (Hybrid 모델) ---
def make_recommendations(user_text, interest=None, region=None, salary=None, shown_companies_set=None, top_n=3):
    if shown_companies_set is None:
        shown_companies_set = set()

    if not user_text or not cached_companies:
        return []

    # TF-IDF 기반 유사도 계산
    tfidf_ranked_companies = tfidf_similarity(user_text, cached_companies)

    results = []
    for company, base_score in tfidf_ranked_companies:
        company_name = company.get("회사명") or company.get("name")
        company_summary = company.get("채용공고명") or company.get("summary")
        
        if not company_name or not company_summary:
            continue
        
        company_key = (company_name, company_summary)
        if company_key in shown_companies_set:
            continue

        boost = 0.0
        if interest and interest.lower() in str(company.get("summary", "")).lower():
            boost += 0.1
        if region and region.lower() in str(company.get("region", "")).lower():
            boost += 0.05
        if salary:
            try:
                salary_int = int(salary)
                min_salary, max_salary = parse_salary_info(company.get("summary", ""))
                if min_salary >= salary_int:
                    boost += 0.05
            except:
                pass

        final_score = base_score + boost
        # 너무 낮은 유사도도 포함시키기 위해 필터 제거
        results.append((company, final_score))

    # 유사도 기준 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    # 상위 top_n개만 반환
    top_results = []
    for comp, sim in results:
        comp_id = (comp.get("name"), comp.get("summary"))
        if comp_id not in shown_companies_set:
            shown_companies_set.add(comp_id)
            top_results.append((comp, sim))
        if len(top_results) >= top_n:
            break

    return top_results

# --- Flask 라우트 설정 ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']

    state = user_states.get(user_id, {
        "shown": set(),
        "user_text": None,
        "interest": None,
        "region": None,
        "salary": None
    })

    message = request.form.get("message", "").strip()
    file = request.files.get("file")

    try:
        # 1. 파일 업로드 시
        if file and file.filename != '':
            user_text = extract_text_from_pdf(file)
            if user_text:
                state["user_text"] = user_text
                state["shown"] = set()
                user_states[user_id] = state
                return jsonify({"reply": "이력서/자기소개서 내용을 확인했습니다. 아래 항목 중에서 관심 분야를 선택해주세요:\n1. IT  2. 제조  3. 서비스  4. 마케팅  5. 상관 없음"})
            else:
                return jsonify({"reply": "PDF 파일에서 텍스트를 추출하는 데 실패했습니다. 다른 파일을 시도해 주세요."})

        # 2. 텍스트 입력 시 자소서로 간주
        if state["user_text"] is None:
            if len(message) > 100 or "이력서" in message or "자기소개서" in message:
                state["user_text"] = message
                state["shown"] = set()
                user_states[user_id] = state
                return jsonify({"reply": "이력서/자기소개서 내용을 확인했습니다. 아래 항목 중에서 관심 분야를 선택해주세요:\n1. IT  2. 제조  3. 서비스  4. 마케팅  5. 상관 없음"})
            else:
                return jsonify({"reply": "자기소개서나 이력서를 먼저 입력해 주세요."})

        # 3. 관심 분야 입력 단계
        if state["interest"] is None:
            option_map = {
                "1": "IT", "2": "제조", "3": "서비스", "4": "마케팅", "5": ""
            }
            selected = option_map.get(message)
            if selected is not None:
                state["interest"] = selected
                user_states[user_id] = state
                display = selected if selected else "상관 없음"
                return jsonify({"reply": f"📌 관심 분야로 '{display}'이(가) 선택되었습니다.\n다음으로 희망 근무지를 선택해주세요:\n1. 서울  2. 경기  3. 인천  4. 부산  5. 상관 없음"})
            else:
                # 자유 텍스트 입력
                state["interest"] = message
                user_states[user_id] = state
                return jsonify({"reply": f"📌 '{message}'을 관심 분야로 설정했습니다.\n다음으로 희망 근무지를 선택해주세요:\n1. 서울  2. 경기  3. 인천  4. 부산  5. 상관 없음"})

        # 4. 희망 근무지 입력 단계
        if state["region"] is None:
            option_map = {
                "1": "서울", "2": "경기", "3": "인천", "4": "부산", "5": ""
            }
            selected = option_map.get(message)
            if selected is not None:
                state["region"] = selected
                user_states[user_id] = state
                display = selected if selected else "상관 없음"
                return jsonify({"reply": f"📌 희망 근무지로 '{display}'이(가) 선택되었습니다.\n다음으로 희망 연봉을 선택해주세요:\n1. 3000~3500만원  2. 3500~4000만원  3. 4000만원 이상  4. 연봉 무관"})
            else:
                # 자유 텍스트 입력
                state["region"] = message
                user_states[user_id] = state
                return jsonify({"reply": f"📌 '{message}'을 희망 근무지로 설정했습니다.\n다음으로 희망 연봉을 선택해주세요:\n1. 3000~3500만원  2. 3500~4000만원  3. 4000만원 이상  4. 연봉 무관"})

        # 5. 희망 연봉 입력 단계
        if state["salary"] is None:
            option_map = {
                "1": "3000", "2": "3500", "3": "4000", "4": ""
            }
            selected = option_map.get(message)
            if selected is not None:
                state["salary"] = selected
            else:
                state["salary"] = message
            user_states[user_id] = state

            interest_text = state["interest"] or "상관 없음"
            region_text = state["region"] or "상관 없음"
            salary_display_map = {
                "3000": "3000~3500만원",
                "3500": "3500~4000만원",
                "4000": "4000만원 이상",
                "": "연봉 무관"
            }
            display = salary_display_map.get(state["salary"], state["salary"])

            summary_msg = (
                f"📌 선택하신 조건은 다음과 같습니다:\n"
                f"- 관심 분야: {interest_text}\n"
                f"- 희망 근무지: {region_text}\n"
                f"- 희망 연봉: {display}\n\n"
                f"이 조건과 자기소개서를 바탕으로 추천 기업을 분석해드릴게요!\n\n"
            )

            # 추천 생성
            new_recommendations = make_recommendations(
                user_text=state["user_text"],
                interest=state.get("interest"),
                region=state.get("region"),
                salary=state.get("salary"),
                shown_companies_set=state["shown"],
                top_n=2
            )

            if not new_recommendations:
                return jsonify({"reply": summary_msg + "현재 조건에 맞는 기업을 찾을 수 없습니다."})

            explanations = []
            for company, score in new_recommendations:
                company_info_for_gpt = {
                    "회사명": company.get("회사명", company.get("name", "정보 없음")),
                    "채용공고명": company.get("채용공고명", company.get("summary", "정보 없음")),
                    "지역": company.get("지역", company.get("region", "")),
                    "산업": company.get("산업", company.get("industry", ""))
                }
                reason = generate_reason_individual(state["user_text"], company_info_for_gpt, score)
                explanations.append(
                    f"**기업명**: {company_info_for_gpt['회사명']}\n"
                    f"**채용공고명**: {company_info_for_gpt['채용공고명']}\n"
                    f"**종합 점수**: {round(score, 2)}\n"
                    f"**설명**: {reason}\n"
                )

            reply = summary_msg + "\n\n".join(explanations)
            reply += "\n\n📌 '더 추천해줘'라고 입력하시면 추가 기업을 추천해드릴게요."
            return jsonify({"reply": reply})

        # 6. 추천 초기화
        if "추천 초기화" in message:
            user_states[user_id] = {
                "shown": set(),
                "user_text": None,
                "interest": None,
                "region": None,
                "salary": None
            }
            return jsonify({"reply": "추천 상태가 초기화되었습니다. 새로운 자기소개서/이력서를 업로드하거나 입력해 주세요."})

        # 7. 기타
        return jsonify({"reply": "무슨 말씀이신지 이해하지 못했습니다. '추천 초기화'로 처음부터 다시 시작할 수 있어요."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        traceback.print_exc()
        return jsonify({"reply": f"❌ 오류가 발생했습니다: {str(e)}"}), 500
        
@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
