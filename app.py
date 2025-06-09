import os
import json
import fitz # PyMuPDF
import openai
import xml.etree.ElementTree as ET
import requests
import pandas as pd

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

# KoBERT 사용을 위한 라이브러리 임포트
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)

# 환경 변수에서 OpenAI API 키 로드 또는 기본값 설정
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
GG_CACHED_FILE = "gg_employment_cached.json"

user_states = {}

# GG 캐시된 데이터 로드 (파일이 존재하지 않으면 에러 발생 가능성 있음)
try:
    with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
        cached_companies = json.load(f)
except FileNotFoundError:
    print(f"경고: '{GG_CACHED_FILE}' 파일을 찾을 수 없습니다. 빈 회사 목록으로 시작합니다.")
    cached_companies = []
except json.JSONDecodeError:
    print(f"오류: '{GG_CACHED_FILE}' 파일이 유효한 JSON 형식이 아닙니다.")
    cached_companies = []


# KoBERT 모델과 토크나이저 로드
# 애플리케이션 시작 시 한 번만 로드하여 효율성을 높입니다.
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert")
    # GPU(CUDA) 사용 가능 시 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 모델을 평가 모드로 설정 (드롭아웃/배치 정규화 비활성화)
    print("KoBERT 모델 로드 성공!")
except Exception as e:
    print(f"❌ KoBERT 모델 로드 오류: {e}")
    print("KoBERT 사용을 위해 'transformers', 'torch', 'sentencepiece' 라이브러리가 설치되어 있는지 확인해주세요.")
    # 모델 로드 실패 시, KoBERT 관련 기능을 사용하지 않거나 애플리케이션을 종료하는 등의 예외 처리가 필요합니다.
    # 여기서는 오류 메시지 출력 후 계속 진행 (KoBERT 관련 함수에서 추가 오류 발생 가능)


# --- PDF에서 텍스트 추출 함수 ---
def extract_text_from_pdf(pdf_file):
    """PDF 파일에서 텍스트를 추출합니다."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text.strip()

# --- GPT를 사용하여 키워드 추출 함수 ---
def extract_keywords(text):
    """GPT 모델을 사용하여 텍스트에서 핵심 키워드를 추출합니다."""
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

# --- KoBERT 임베딩 생성 함수 ---
def get_kobert_embedding(text):
    """주어진 텍스트에 대한 KoBERT 임베딩(벡터)을 생성합니다."""
    if not text:
        # 빈 텍스트인 경우 KoBERT 모델의 hidden_size와 동일한 크기의 0 벡터 반환
        return torch.zeros(model.config.hidden_size).to(device)
    try:
        # 텍스트를 토큰화하고 PyTorch 텐서로 변환
        # max_length=512는 KoBERT의 최대 입력 길이에 맞춥니다.
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # 입력을 모델이 있는 장치(CPU/GPU)로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(): # 기울기 계산 비활성화 (메모리 절약, 속도 향상)
            outputs = model(**inputs)

        # 마지막 은닉 상태(Last Hidden State)의 평균을 문장 임베딩으로 사용
        # outputs.last_hidden_state의 형태는 (batch_size, sequence_length, hidden_size)
        # dim=1로 평균을 내어 (batch_size, hidden_size) 형태로 만듭니다.
        # squeeze()를 통해 차원 1을 제거하여 (hidden_size,) 형태로 만듭니다.
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
        return embedding
    except Exception as e:
        print(f"❌ KoBERT 임베딩 생성 오류 (텍스트 앞부분: '{text[:50]}...'): {e}")
        # 오류 발생 시 0 벡터 반환하여 프로그램 중단 방지
        return torch.zeros(model.config.hidden_size).to(device)


# --- KoBERT를 사용한 유사도 계산 함수 ---
def kobert_similarity(user_text, companies):
    """
    사용자 텍스트와 회사 요약 간의 KoBERT 기반 코사인 유사도를 계산하고 정렬하여 반환합니다.
    """
    if not user_text:
        return []

    user_embedding = get_kobert_embedding(user_text)
    # NumPy 배열로 변환 및 sklearn cosine_similarity 함수를 위한 차원 조정
    user_embedding_np = user_embedding.cpu().numpy().reshape(1, -1)

    company_data_with_summaries = []
    for c in companies:
        summary = f"{c.get('채용공고명', '')} {c.get('회사명', '')}"
        company_data_with_summaries.append((c, summary))

    if not company_data_with_summaries:
        return []

    # 모든 회사 요약에 대한 임베딩을 일괄적으로 계산 (성능 향상)
    company_summaries = [item[1] for item in company_data_with_summaries]
    
    # KoBERT 임베딩을 생성할 때 빈 문자열이 있으면 torch.stack에서 오류가 발생할 수 있으므로,
    # 유효한 임베딩만 스택하도록 필터링합니다.
    company_embeddings_list = [get_kobert_embedding(s) for s in company_summaries]
    
    # 유효한 임베딩만 필터링 (get_kobert_embedding에서 0 벡터를 반환했는지 확인)
    valid_embeddings_and_indices = []
    for i, emb in enumerate(company_embeddings_list):
        # 0 벡터가 아니거나, 0 벡터여도 실제 길이가 있는 경우만 유효하다고 판단
        if not torch.all(emb == 0) or len(company_summaries[i]) > 0:
            valid_embeddings_and_indices.append((emb, i))
    
    if not valid_embeddings_and_indices:
        return []

    # 유효한 임베딩들만 스택
    valid_company_embeddings_tensor = torch.stack([item[0] for item in valid_embeddings_and_indices])
    # CPU로 옮긴 후 NumPy 배열로 변환
    valid_company_embeddings_np = valid_company_embeddings_tensor.cpu().numpy()

    # 코사인 유사도 계산 (사용자 임베딩과 모든 유효한 회사 임베딩 간)
    cosine_sim_scores = cosine_similarity(user_embedding_np, valid_company_embeddings_np).flatten()

    results = []
    # 유사도 점수와 원본 회사 정보를 매핑
    for i, score in enumerate(cosine_sim_scores):
        original_index = valid_embeddings_and_indices[i][1]
        results.append((company_data_with_summaries[original_index][0], score))

    # 유사도 점수를 기준으로 내림차순 정렬
    return sorted(results, key=lambda x: x[1], reverse=True)


# --- GPT를 사용하여 추천 이유 생성 함수 ---
def generate_reason_individual(user_text, company, score):
    """GPT 모델을 사용하여 특정 기업에 대한 매칭 이유를 생성합니다."""
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

# --- 기업 추천 로직 함수 ---
def make_recommendations(user_text, interest, region, salary, shown=set(), top_n=3):
    """
    사용자 텍스트와 선호도(관심 분야, 지역, 연봉)를 기반으로 기업을 추천합니다.
    """
    if not user_text:
        return []

    # 1. GPT를 이용해 자기소개서/이력서에서 핵심 키워드 추출
    keywords = extract_keywords(user_text)

    # 2. 키워드, 관심 분야, 지역, 연봉 등을 고려한 초기 스코어링
    def score_company(company):
        s = 0
        summary = company.get("채용공고명", "") + " " + company.get("회사명", "")
        # 추출된 키워드가 회사 요약에 포함되면 점수 부여
        if any(kw in summary for kw in keywords):
            s += 1
        # 관심 분야가 회사 요약에 포함되면 점수 부여
        if interest and interest in summary:
            s += 0.3
        # 희망 근무 지역이 회사 정보에 포함되면 점수 부여
        if region and region in company.get("근무지역", ""):
            s += 0.3
        # 희망 연봉이 회사 급여 정보에 포함되면 점수 부여 (간단한 문자열 일치)
        if salary and str(salary) in company.get("급여", ""):
            s += 0.2
        return s

    # 초기 스코어링을 기반으로 회사 목록을 정렬하여 KoBERT 유사도 계산의 후보군으로 사용
    filtered_companies = sorted(cached_companies, key=score_company, reverse=True)
    
    # 3. KoBERT 유사도를 사용하여 최종 순위 결정
    # 이미 추천된 기업을 제외하고 KoBERT 유사도 계산을 수행
    # Kobert_similarity 함수는 유사도에 따라 정렬된 (회사, 유사도 점수) 튜플 리스트를 반환합니다.
    kobert_ranked_companies = kobert_similarity(user_text, filtered_companies)
    
    results = []
    for comp, sim in kobert_ranked_companies:
        # 유사도 점수가 0보다 크고, 이미 추천 목록에 없는 기업만 추가
        if sim > 0.0 and (comp.get("회사명"), comp.get("채용공고명")) not in shown:
            shown.add((comp.get("회사명"), comp.get("채용공고명"))) # 추천된 기업을 shown set에 추가
            results.append((comp, sim))
        if len(results) >= top_n: # 요청된 개수만큼 추천되면 중단
            break
            
    return results

# --- Flask 라우트 설정 ---
@app.route("/")
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """사용자와의 채팅 상호작용을 처리합니다."""
    user_id = request.remote_addr # 사용자 식별 (IP 주소 사용)
    message = request.form.get("message", "").strip() # 사용자 메시지
    file = request.files.get("file") # 첨부된 파일 (PDF)
    state = user_states.get(user_id, {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None}) # 사용자 상태 관리 딕셔너리

    try:
        # 1. PDF 파일 업로드 처리
        if file:
            user_text = extract_text_from_pdf(file)
            state["user_text"] = user_text
            user_states[user_id] = state # 상태 업데이트
            return jsonify({"reply": "감사합니다. 이력서/자기소개서 내용이 성공적으로 분석되었습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 입력해 주세요. (예시: 품질, 서울, 3000만원 또는 없음, 없음, 없음)"})

        # 2. 파일 없이 사용자가 이력서/자기소개서 내용을 직접 메시지로 보낸 경우
        # user_text가 아직 없고, 메시지가 이력서/자소서로 판단될 만큼 길거나 특정 키워드를 포함할 때
        if state["user_text"] is None and message:
            # 간단한 휴리스틱: 메시지 길이가 길거나 "이력서", "자기소개서" 키워드 포함 시
            if len(message.split()) > 30 or "이력서" in message or "자기소개서" in message:
                state["user_text"] = message
                user_states[user_id] = state
                return jsonify({"reply": "이력서/자기소개서 내용을 확인했습니다. 이제 **관심 분야, 희망 근무지, 연봉**을 입력해 주세요. (예시: 품질, 서울, 3000만원 또는 없음, 없음, 없음)"})
            # 일반 대화로 간주
            else:
                return jsonify({"reply": "개인별 맞춤 분석을 위해서는 자기소개서 혹은 이력서가 필요합니다. 파일을 첨부해 주시거나 내용을 직접 입력해 주시면 상세한 분석을 제공해드리겠습니다."})


        # 3. 사용자 선호도 정보 (관심 분야, 희망 근무지, 연봉) 입력 처리
        # user_text가 이미 있고, 관심 분야, 지역, 연봉 정보가 아직 설정되지 않았을 때
        if state["user_text"] is not None and state["interest"] is None and "," in message:
            parts = [p.strip() for p in message.split(",")]
            state["interest"] = parts[0] if len(parts) > 0 and parts[0] != "없음" else ""
            state["region"] = parts[1] if len(parts) > 1 and parts[1] != "없음" else ""
            state["salary"] = parts[2].replace("만원", "") if len(parts) > 2 and parts[2] != "없음" else ""
            user_states[user_id] = state

            # 선호도 정보 입력이 완료되면 즉시 추천 시작 (기본 3개)
            new_recommendations = make_recommendations(
                user_text=state["user_text"],
                interest=state.get("interest"),
                region=state.get("region"),
                salary=state.get("salary"),
                shown=state["shown"],
                top_n=3
            )

            if not new_recommendations:
                return jsonify({"reply": "아쉽게도 현재 조건에 맞는 새로운 기업을 찾을 수 없습니다. 다른 조건을 말씀해주시겠어요?"})

            explanations = []
            for company, score in new_recommendations:
                reason = generate_reason_individual(state["user_text"], company, score)
                explanations.append(f"**기업명**: {company.get('회사명', '정보 없음')}\n**채용공고명**: {company.get('채용공고명', '정보 없음')}\n**유사도 점수**: {round(score,2)}\n**설명**: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 예를 들어 '더 추천해줘'라고 말씀하시면 다른 기업을 찾아드릴 수 있습니다."
            return jsonify({"reply": reply})

        # 4. 추가 추천 요청 처리 ("더 추천해줘" 메시지)
        # 이미 user_text와 선호도 정보가 설정되어 있고, 사용자가 추가 추천을 요청했을 때
        if state["user_text"] is not None and state["interest"] is not None and "더 추천해줘" in message:
            new_recommendations = make_recommendations(
                user_text=state["user_text"],
                interest=state.get("interest"),
                region=state.get("region"),
                salary=state.get("salary"),
                shown=state["shown"],
                top_n=1 # 한 번에 하나씩만 추가 추천
            )

            if not new_recommendations:
                return jsonify({"reply": "더 이상 추천할 기업이 없습니다. 다른 조건을 말씀해주시겠어요?"})

            explanations = []
            for company, score in new_recommendations:
                reason = generate_reason_individual(state["user_text"], company, score)
                explanations.append(f"**기업명**: {company.get('회사명', '정보 없음')}\n**채용공고명**: {company.get('채용공고명', '정보 없음')}\n**유사도 점수**: {round(score,2)}\n**설명**: {reason}\n")

            reply = "\n\n".join(explanations)
            reply += "\n\n📌 더 궁금한 점이나 고려하고 싶은 조건이 있다면 말씀해 주세요. 추가로 반영해 드릴게요! 또는 '추천 초기화'라고 말씀하시면 처음부터 다시 시작할 수 있습니다."
            return jsonify({"reply": reply})
        
        # 5. 추천 초기화 요청 (새로운 PDF/자소서로 시작)
        if "추천 초기화" in message:
            user_states[user_id] = {"shown": set(), "user_text": None, "interest": None, "region": None, "salary": None}
            return jsonify({"reply": "추천 상태가 초기화되었습니다. 새로운 자기소개서/이력서 파일을 첨부하시거나 내용을 직접 입력해 주세요."})


        # 모든 조건에 해당하지 않을 경우
        return jsonify({"reply": "무슨 말씀이신지 정확히 이해하지 못했습니다. 자기소개서/이력서를 첨부해 주시거나, '추천 초기화'를 통해 다시 시작할 수 있습니다."})

    except Exception as e:
        print(f"❌ 서버 에러: {e}")
        # 예외 발생 시 사용자에게 오류 메시지 반환
        return jsonify({"reply": f"❌ 오류가 발생했습니다: {str(e)} 불편을 드려 죄송합니다. 잠시 후 다시 시도해 주세요."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
