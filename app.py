import os
import json
import fitz # PyMuPDF
import openai
import uuid
import re # 정규표현식 모듈 추가
import traceback # 오류 스택 트레이스 출력을 위한 모듈 추가

import torch
from transformers import AutoTokenizer, AutoModel

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
# from werkzeug.utils import secure_filename # 사용하지 않으므로 제거

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)
# 세션 관리를 위한 secret_key 설정 (반드시 실제 환경에서는 강력한 무작위 문자열로 변경해야 함!)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-to-a-super-secret-random-string")

# --- API 키 로딩 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 프로그램을 시작할 수 없습니다.")
openai.api_key = api_key

GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {} # 사용자별 대화 상태를 저장할 딕셔너리 (인메모리)

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

    # KoBERT 모델과 토크나이저 로드 (전역 변수로 설정)
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("✅ KoBERT 모델 로드 성공!")

    # --- 서버 시작 시 모든 기업 정보에 대한 KoBERT 임베딩 미리 계산 ---
    print("서버 시작 전, 기업 정보 KoBERT 임베딩을 시작합니다... (데이터 양에 따라 몇 분 소요될 수 있습니다)")
    def get_kobert_embedding_for_startup(text_input):
        if not text_input:
            # 텍스트가 비어있을 경우 0으로 채워진 임베딩 반환
            return torch.zeros(model.config.hidden_size).to(device)
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
            return torch.zeros(model.config.hidden_size).to(device)

    for company in cached_companies:
        # '채용공고명'과 '회사명'을 조합하여 임베딩 생성
        summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        company['embedding'] = get_kobert_embedding_for_startup(summary)
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
except Exception as e:
    print(f"❌ 초기 설정 중 치명적인 오류 발생: {e}")
    traceback.print_exc()
    # 필수 모듈 로딩 실패 시 앱 시작을 막습니다.
    raise RuntimeError("애플리케이션 초기 설정에 실패했습니다. 필수 모델 및 데이터 로딩 오류.")


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
    if model is None or tokenizer is None or device is None or not text_input:
        # KoBERT 모델이 로드되지 않았거나 입력 텍스트가 없을 경우 0 벡터 반환
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


# --- 기업 추천 로직 함수 (Hybrid 모델) ---
def make_recommendations(user_text, interest, region, salary, shown_companies_set=None, top_n=3):
    if shown_companies_set is None:
        shown_companies_set = set()

    if not user_text:
        return []

    # 1. TF-IDF 기반으로 1차 필터링하여 후보 기업군 선정
    # (tfidf_similarity 함수가 전체 cached_companies에 대해 동작하도록 수정 필요)
    # 현재 tfidf_similarity는 이미 make_recommendations 내에서 호출됩니다.
    # make_recommendations 내부의 tfidf_scores를 생성하는 로직은 이미 전체 cached_companies를 대상으로 합니다.

    # 2. KoBERT 기반 유사도 계산 (전체 캐싱된 기업 대상)
    kobert_ranked_companies = kobert_similarity(user_text, cached_companies)

    # 3. TF-IDF 기반 유사도 계산 (전체 캐싱된 기업 대상)
    tfidf_ranked_companies = tfidf_similarity(user_text, cached_companies)
    
    # 4. 점수 합산을 위해 회사 정보를 key로 하는 딕셔너리 생성
    tfidf_scores = {
        (c.get("회사명"), c.get("채용공고명")): score
        for c, score in tfidf_ranked_companies
    }

    # 5. Hybrid 점수 계산 (KoBERT 점수 + TF-IDF 점수)
    hybrid_scores = []
    for company, kobert_score in kobert_ranked_companies:
        company_key = (company.get("회사명"), company.get("채용공고명"))
        tfidf_score = tfidf_scores.get(company_key, 0.0)

        # 가중치 설정 (KoBERT: 70%, TF-IDF: 30%) - 이 값은 조정 가능
        kobert_weight = 0.7
        tfidf_weight = 0.3
        
        # 점수 정규화 (필요시) - 현재는 코사인 유사도 자체가 0~1이므로 큰 문제는 없지만,
        # TF-IDF 스코어가 너무 낮게 나오는 경우 가중치 조절이 중요
        final_score = (kobert_weight * kobert_score) + (tfidf_weight * tfidf_score)
        
        # 추가 필터링 (관심분야, 지역 등)
        passes_filter = True
        summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        
        # '관심' 키워드가 채용공고명/회사명에 포함되는지 확인
        if interest and interest.lower() not in summary.lower():
            passes_filter = False
        
        # '지역' 키워드가 근무지역에 포함되는지 확인
        if region and region.lower() not in company.get("근무지역", "").lower():
            passes_filter = False
        
        # 연봉 필터링은 현재 구현되어 있지 않으므로, 필요시 여기에 추가
        # if salary:
        #    try:
        #        min_salary_req = int(salary)
        #        company_salary_info = company.get("연봉정보", "") # 실제 데이터 키에 따라 변경
        #        # ... 연봉 정보 파싱 및 비교 로직 추가 ...
        #    except ValueError:
        #        pass # 유효하지 않은 연봉 입력은 무시

        if passes_filter:
            hybrid_scores.append((company, final_score))

    # 최종 점수를 기준으로 내림차순 정렬
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 이미 보여준 공고 제외하고 상위 N개 선택
    results = []
    for comp, sim in hybrid_scores:
        comp_id = (comp.get("회사명"), comp.get("채용공고명"))
        if comp_id not in shown_companies_set:
            shown_companies_set.add(comp_id) # 이미 보여준 회사 목록에 추가
            results.append((comp, sim))
        if len(results) >= top_n:
            break
            
    return results


# --- Flask 라우트 설정 ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # 사용자 ID가 세션에 없으면 새로 생성
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    
    # user_states 딕셔너리에서 현재 사용자 상태 로드
    state = user_states.get(user_id, {
        "shown": set(), # 이미 추천된 기업 목록 (중복 추천 방지)
        "user_text": None, # 사용자의 자기소개서/이력서 텍스트
        "interest": None, # 사용자 관심 분야
        "region": None, # 사용자 희망 근무지
        "salary": None # 사용자 희망 연봉
    })

    message = request.form.get("message", "").strip()
    file = request.files.get("file")

    try:
        # 1. 파일 첨부 시 자기소개서/이력서 추출
        if file and file.filename != '':
            user_text = extract_text_from_pdf(file)
            if user_text:
                state["user_text"] = user_text
                # 파일 업로드 시에는 기존 추천 이력을 초기화
                state["shown"] = set()
                user_states[user_id] = state
                return jsonify({"reply": "감사합니다. 이력서/자기소개서 내용이 성공적으로 분석되었습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 입력해 주세요."})
            else:
                return jsonify({"reply": "PDF 파일에서 텍스트를 추출하는 데 실패했습니다. 파일이 스캔된 이미지 기반이거나 손상되었을 수 있습니다. 텍스트를 직접 입력해 주시거나 다른 파일을 시도해 주시겠어요?"})

        # 2. 파일 없이 메시지만 있을 경우
        if state["user_text"] is None:
            # 메시지 길이가 길거나 특정 키워드를 포함하면 자기소개서/이력서로 간주
            if len(message.split()) > 30 or "이력서" in message or "자기소개서" in message:
                state["user_text"] = message
                state["shown"] = set() # 텍스트 직접 입력 시에도 기존 추천 이력 초기화
                user_states[user_id] = state
                return jsonify({"reply": "이력서/자기소개서 내용을 확인했습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 입력해 주세요."})
            else:
                # 자기소개서/이력서가 없는 경우 일반 상담 모드 메시지
                return jsonify({"reply": "개인별 맞춤 분석을 위해서는 자기소개서 혹은 이력서가 필요합니다. 파일을 첨부해 주시거나 내용을 직접 입력해 주시면 상세한 분석을 제공해 드리겠습니다."})

        # 3. 자기소개서/이력서가 입력되었고, 사용자 선호도 정보가 없는 경우
        if state["user_text"] is not None and state["interest"] is None:
            if "," in message:
                parts = [p.strip() for p in message.split(",")]
                state["interest"] = parts[0] if len(parts) > 0 and parts[0].lower() != "없음" else ""
                state["region"] = parts[1] if len(parts) > 1 and parts[1].lower() != "없음" else ""
                state["salary"] = parts[2].replace("만원", "") if len(parts) > 2 and parts[2].lower() != "없음" else ""
                user_states[user_id] = state

                # 선호도 입력 후 첫 추천 시작
                new_recommendations = make_recommendations(
                    user_text=state["user_text"],
                    interest=state.get("interest"),
                    region=state.get("region"),
                    salary=state.get("salary"),
                    shown_companies_set=state["shown"],
                    top_n=3 # 첫 추천은 3개
                )

                if not new_recommendations:
                    return jsonify({"reply": "아쉽게도 현재 조건에 맞는 기업을 찾을 수 없습니다. 다른 조건을 말씀해주시거나 '추천 초기화'를 통해 다시 시작해 주시겠어요?"})

                explanations = []
                for company, score in new_recommendations:
                    # 임베딩 정보는 클라이언트에게 보낼 필요 없으므로 제거
                    company_info_for_gpt = {k: v for k, v in company.items() if k != 'embedding'}
                    reason = generate_reason_individual(state["user_text"], company_info_for_gpt, score)
                    explanations.append(f"**기업명**: {company_info_for_gpt.get('회사명', '정보 없음')}\n**채용공고명**: {company_info_for_gpt.get('채용공고명', '정보 없음')}\n**종합 점수**: {round(score,2)}\n**설명**: {reason}\n")

                reply = "\n\n".join(explanations)
                reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 예를 들어 '더 추천해줘'라고 말씀하시면 다른 기업을 찾아드릴 수 있습니다."
                return jsonify({"reply": reply})
            else:
                return jsonify({"reply": "관심 분야, 희망 근무지, 연봉을 '품질, 서울, 3000만원' 또는 '없음, 없음, 없음'과 같이 콤마(,)로 구분해서 입력해 주세요."})

        # 4. "더 추천해줘" 요청 처리
        if state["user_text"] is not None and state["interest"] is not None and "더 추천해줘" in message:
            new_recommendations = make_recommendations(
                user_text=state["user_text"],
                interest=state.get("interest"),
                region=state.get("region"),
                salary=state.get("salary"),
                shown_companies_set=state["shown"],
                top_n=1 # 추가 추천은 1개씩
            )

            if not new_recommendations:
                return jsonify({"reply": "더 이상 추천할 기업이 없습니다. 다른 조건을 말씀해주시거나 '추천 초기화'를 통해 다시 시작해 주시겠어요?"})

            explanations = []
            for company, score in new_recommendations:
                company_info_for_gpt = {k: v for k, v in company.items() if k != 'embedding'}
                reason = generate_reason_individual(state["user_text"], company_info_for_gpt, score)
                explanations.append(f"**기업명**: {company_info_for_gpt.get('회사명', '정보 없음')}\n**채용공고명**: {company_info_for_gpt.get('채용공고명', '정보 없음')}\n**종합 점수**: {round(score,2)}\n**설명**: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 또는 '추천 초기화'라고 말씀하시면 처음부터 다시 시작할 수 있습니다."
            return jsonify({"reply": reply})
            
        # 5. "추천 초기화" 요청 처리
        if "추천 초기화" in message:
            user_states[user_id] = {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None}
            return jsonify({"reply": "추천 상태가 초기화되었습니다. 새로운 자기소개서/이력서 파일을 첨부하시거나 내용을 직접 입력해 주세요."})

        # 기타 일반 메시지 처리 (자기소개서/이력서 및 선호도 입력 후)
        # 여기에 일반적인 질의응답 로직을 추가할 수 있습니다.
        # 예: 기업 정보를 묻는 질문, 면접 관련 질문 등.
        # 현재는 이 단계에서 다른 질문이 오면 이해하지 못한다고 응답합니다.
        return jsonify({"reply": "무슨 말씀이신지 정확히 이해하지 못했습니다. 자기소개서/이력서를 첨부해 주시거나, '추천 초기화'를 통해 다시 시작할 수 있습니다."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        traceback.print_exc() # 서버 전체 오류 스택 트레이스 출력
        return jsonify({"reply": f"❌ 오류가 발생했습니다: {str(e)} 불편을 드려 죄송합니다. 잠시 후 다시 시도해 주세요."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
