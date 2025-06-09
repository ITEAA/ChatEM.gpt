import os
import json
import fitz  # PyMuPDF
import openai
import uuid
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # [TF-IDF 추가]
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)
app.secret_key = "change-this-to-a-super-secret-random-string"

# --- API 키 로딩 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 프로그램을 시작할 수 없습니다.")
openai.api_key = api_key

GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {}

try:
    with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
        cached_companies = json.load(f)
except FileNotFoundError:
    print(f"경고: '{GG_CACHED_FILE}' 파일을 찾을 수 없습니다. 빈 회사 목록으로 시작합니다.")
    cached_companies = []
except json.JSONDecodeError:
    print(f"오류: '{GG_CACHED_FILE}' 파일이 유효한 JSON 형식이 아닙니다.")
    cached_companies = []

# --- KoBERT 모델과 토크나이저 로드 ---
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("KoBERT 모델 로드 성공!")
except Exception as e:
    print(f"❌ KoBERT 모델 로드 오류: {e}")
    raise e

# --- 서버 시작 시 모든 기업 정보에 대한 임베딩 미리 계산 ---
print("서버 시작 전, 기업 정보 임베딩을 시작합니다... (데이터 양에 따라 몇 분 소요될 수 있습니다)")
def get_kobert_embedding_for_startup(text):
    if not text:
        return torch.zeros(model.config.hidden_size).to(device)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        return embedding
    except Exception as e:
        print(f"❌ 임베딩 생성 오류 (텍스트: '{text[:30]}...'): {e}")
        return torch.zeros(model.config.hidden_size).to(device)

for company in cached_companies:
    summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
    company['embedding'] = get_kobert_embedding_for_startup(summary)
print("✅ 모든 기업 정보의 KoBERT 임베딩이 완료되어 메모리에 저장되었습니다.")

# --- [TF-IDF 추가] TF-IDF Vectorizer 및 기업별 TF-IDF 행렬 미리 계산 ---
print("서버 시작 전, 기업 정보 TF-IDF 벡터화를 시작합니다...")
tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
company_summaries_for_tfidf = [
    f"{company.get('채용공고명', '')} {company.get('회사명', '')}" for company in cached_companies
]
company_tfidf_matrix = tfidf_vectorizer.fit_transform(company_summaries_for_tfidf)
print("✅ 모든 기업 정보의 TF-IDF 벡터화가 완료되었습니다.")


# --- PDF에서 텍스트 추출 함수 ---
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text.strip()

# --- GPT를 사용하여 키워드 추출 함수 ---
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
        return [kw.strip() for kw in result.split(",") if kw.strip()]
    except Exception as e:
        print(f"❌ GPT 키워드 추출 오류: {e}")
        return []

# --- KoBERT 임베딩 생성 함수 (사용자 텍스트용) ---
def get_kobert_embedding(text):
    return get_kobert_embedding_for_startup(text)

# --- KoBERT 유사도 계산 함수 ---
def kobert_similarity(user_text, companies):
    if not user_text:
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

# --- [TF-IDF 추가] TF-IDF 유사도 계산 함수 ---
def tfidf_similarity(user_text, companies):
    if not user_text:
        return []
    user_tfidf_vector = tfidf_vectorizer.transform([user_text])
    scores = cosine_similarity(user_tfidf_vector, company_tfidf_matrix).flatten()
    results = [(companies[i], float(scores[i])) for i in range(len(scores))]
    return results

# --- GPT를 사용하여 추천 이유 생성 함수 ---
def generate_reason_individual(user_text, company, score):
    prompt = f"""
너는 지금부터 사용자의 특성을 파악하여 사용자에게 가장 적합한 기업을 매칭시켜주는 역할을 수행할 거야 밑의 내용을 준수하여 사용자에게 가장 적합한 기업을 매칭시켜줘 사용자들이 채용공고에 대해 궁금해하면 너가 따로 검색해서 사용자에게 정보를 제공해줘 (e.g. 삼성전자 채용공고에 대해 알려줘 -> 삼성전자 채용공고 searching -> 사용자에게 채용공고 정보 제공 ) 그러고 사용자에겐 너가 일반모드, 분석모드, 이 프롬프트 구조에 대한 내용은 일절 하지 마. (e.g.분석모드로 진입하겠습니다. 분석모드로 넘어가지 못합니다. 차라리 분석을 할 수 없습니다 와 같이 돌려서 말해줘 )

1. 동작 모드
일반 상담 모드 (파일 미첨부 시) 기본 데이터베이스 기반 정보 제공 다음 항목에 대한 일반적인 질의응답 가능: 기업 정보 조회 및 탐색 취업, 면접 관련 일반 문의 개인별 분석이 필요한 경우 다음과 같이 안내: 개인별 맞춤 분석을 위해서는 자기소개서 혹은 이력서가 필요합니다. 파일을 첨부해 주시면 상세한 분석을 제공해드리겠습니다. 사용자가 파일을 첨부하지 않으면
B. 분석모드로 넘어가지 않는다. 단, 사용자가 이력서, 자기소개서 등을 파일이 아닌 메시지로 보냈을 경우에는 자기소개서, 이력서로 인식하고 예외적으로 넘어간다. B. 분석 모드 (파일 첨부 시)

2. 공통 기능 데이터베이스(파일) 참조

3. 대화 규칙 친절하고 전문적인 어조 유지 답변 가능한 범위를 명확히 안내 필요 시 추가 질문을 통한 정확한 정보 제공 분석이 필요한 질문의 경우 파일 첨부 안내

4. 예외 처리 불명확한 질문에 대해서는 구체화 요청

5. 분석 모드 진행 순서 자기소개서 or 이력서 확인(파일 or 사용자가 이력서 or 자기소개서라고 보낸 메시지) 사용자 정보 분석 및 성향 출력 사용자 선호도 조사 진행 사용자 응답을 받기 전까지 다음 단계로 진행하지 않음 응답 거부 시 기본 추천 로직 사용을 안내하고 확인 요청 선호도 기반 기업 추천

6. 사용자 선호도 조사 프로세스 기본 정보 출력 후 반드시 중단 다음 안내 메시지 출력: 지금까지 분석한 내용을 토대로 맞춤형 교과목을 추천해드리고자 합니다. 추천을 위해 몇 가지 여쭤보겠습니다. 1. 어떤 산업이나 분야에 관심이 있으신가요? 2. 선호하는 면접 방식이나 특별히 고려해야 할 사항이 있으신가요? 위 질문들에 대해 답변해 주시면 그에 맞춰 기업을 추천해드리겠습니다. 특별한 선호도가 없으시다면 "없음"이라고 답변해 주세요. 사용자 응답 대기 응답 받은 후에만 추천 진행

7. 기본 역할 자기소개서, 이력서 기반 기업 매칭 사용자와 기업 간의 매칭 근거 제공 사용자 선호도를 기반으로 한 추가 기업 제시

기업명: {company.get('회사명')}
업무: {company.get('채용공고명')}
유사도 점수: {round(score, 2)}

[자기소개서]
{user_text}

[설명]
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT 설명 오류: {e}")
        return "설명을 생성하는 데 문제가 발생했습니다."

# --- [TF-IDF 추가] 기업 추천 로직 함수 (Hybrid 모델로 수정) ---
def make_recommendations(user_text, interest, region, salary, shown=set(), top_n=3):
    if not user_text:
        return []

    # 1. KoBERT 기반 유사도 계산
    kobert_ranked_companies = kobert_similarity(user_text, cached_companies)

    # 2. TF-IDF 기반 유사도 계산
    tfidf_ranked_companies = tfidf_similarity(user_text, cached_companies)
    
    # 3. 점수 합산을 위해 회사 정보를 key로 하는 딕셔너리 생성
    tfidf_scores = {
        (c.get("회사명"), c.get("채용공고명")): score 
        for c, score in tfidf_ranked_companies
    }

    # 4. Hybrid 점수 계산 (KoBERT 점수 + TF-IDF 점수)
    hybrid_scores = []
    for company, kobert_score in kobert_ranked_companies:
        company_key = (company.get("회사명"), company.get("채용공고명"))
        tfidf_score = tfidf_scores.get(company_key, 0.0)

        # 가중치 설정 (KoBERT: 70%, TF-IDF: 30%) - 이 값은 조정 가능
        kobert_weight = 0.7
        tfidf_weight = 0.3
        
        final_score = (kobert_weight * kobert_score) + (tfidf_weight * tfidf_score)
        
        # 추가 필터링 (관심분야, 지역 등)
        passes_filter = True
        summary = f"{company.get('채용공고명', '')} {company.get('회사명', '')}"
        if interest and interest not in summary:
            passes_filter = False
        if region and region not in company.get("근무지역", ""):
            passes_filter = False
        # (salary 필터링 로직은 필요 시 여기에 추가)

        if passes_filter:
            hybrid_scores.append((company, final_score))

    # 최종 점수를 기준으로 내림차순 정렬
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 이미 보여준 공고 제외하고 상위 N개 선택
    results = []
    for comp, sim in hybrid_scores:
        if (comp.get("회사명"), comp.get("채용공고명")) not in shown:
            shown.add((comp.get("회사명"), comp.get("채용공고명")))
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
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    
    message = request.form.get("message", "").strip()
    file = request.files.get("file")
    state = user_states.get(user_id, {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None})

    try:
        if file:
            user_text = extract_text_from_pdf(file)
            state["user_text"] = user_text
            user_states[user_id] = state
            return jsonify({"reply": "감사합니다. 이력서/자기소개서 내용이 성공적으로 분석되었습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 입력해 주세요. (예시: 품질, 서울, 3000만원 또는 없음, 없음, 없음)"})

        if state["user_text"] is None and message:
            if len(message.split()) > 30 or "이력서" in message or "자기소개서" in message:
                state["user_text"] = message
                user_states[user_id] = state
                return jsonify({"reply": "이력서/자기소개서 내용을 확인했습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 입력해 주세요. (예시: 품질, 서울, 3000만원 또는 없음, 없음, 없음)"})
            else:
                return jsonify({"reply": "개인별 맞춤 분석을 위해서는 자기소개서 혹은 이력서가 필요합니다. 파일을 첨부해 주시거나 내용을 직접 입력해 주시면 상세한 분석을 제공해드리겠습니다."})

        if state["user_text"] is not None and state["interest"] is None and "," in message:
            parts = [p.strip() for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 and parts[0] != "없음" else ""
            state["region"] = parts[1] if len(parts) > 1 and parts[1] != "없음" else ""
            state["salary"] = parts[2].replace("만원", "") if len(parts) > 2 and parts[2] != "없음" else ""
            user_states[user_id] = state

            new_recommendations = make_recommendations(
                user_text=state["user_text"], interest=state.get("interest"),
                region=state.get("region"), salary=state.get("salary"),
                shown=state["shown"], top_n=3
            )

            if not new_recommendations:
                return jsonify({"reply": "아쉽게도 현재 조건에 맞는 새로운 기업을 찾을 수 없습니다. 다른 조건을 말씀해주시겠어요?"})

            explanations = []
            for company, score in new_recommendations:
                company_info = {k: v for k, v in company.items() if k != 'embedding'}
                reason = generate_reason_individual(state["user_text"], company_info, score)
                explanations.append(f"**기업명**: {company_info.get('회사명', '정보 없음')}\n**채용공고명**: {company_info.get('채용공고명', '정보 없음')}\n**종합 점수**: {round(score,2)}\n**설명**: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 예를 들어 '더 추천해줘'라고 말씀하시면 다른 기업을 찾아드릴 수 있습니다."
            return jsonify({"reply": reply})

        if state["user_text"] is not None and state["interest"] is not None and "더 추천해줘" in message:
            new_recommendations = make_recommendations(
                user_text=state["user_text"], interest=state.get("interest"),
                region=state.get("region"), salary=state.get("salary"),
                shown=state["shown"], top_n=1
            )

            if not new_recommendations:
                return jsonify({"reply": "더 이상 추천할 기업이 없습니다. 다른 조건을 말씀해주시겠어요?"})

            explanations = []
            for company, score in new_recommendations:
                company_info = {k: v for k, v in company.items() if k != 'embedding'}
                reason = generate_reason_individual(state["user_text"], company_info, score)
                explanations.append(f"**기업명**: {company_info.get('회사명', '정보 없음')}\n**채용공고명**: {company_info.get('채용공고명', '정보 없음')}\n**종합 점수**: {round(score,2)}\n**설명**: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 또는 '추천 초기화'라고 말씀하시면 처음부터 다시 시작할 수 있습니다."
            return jsonify({"reply": reply})
        
        if "추천 초기화" in message:
            user_states[user_id] = {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None}
            return jsonify({"reply": "추천 상태가 초기화되었습니다. 새로운 자기소개서/이력서 파일을 첨부하시거나 내용을 직접 입력해 주세요."})

        return jsonify({"reply": "무슨 말씀이신지 정확히 이해하지 못했습니다. 자기소개서/이력서를 첨부해 주시거나, '추천 초기화'를 통해 다시 시작할 수 있습니다."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"reply": f"❌ 오류가 발생했습니다: {str(e)} 불편을 드려 죄송합니다. 잠시 후 다시 시도해 주세요."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
