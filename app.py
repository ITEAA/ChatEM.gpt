from flask import Flask, request, jsonify, render_template
import os, re, requests, xml.etree.ElementTree as ET
from dotenv import load_dotenv
from functools import lru_cache
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
assistant_id = os.getenv("ASSISTANT_ID")
job_api_key = os.getenv("JOB_API_KEY")

SYSTEM_PROMPT = """
너는 지금부터 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 추천해주는 역할을 수행한다. 다음 지침을 따라:

1. 동작 모드
- [일반 상담 모드]: 사용자가 자기소개서/이력서를 보내지 않은 경우
  - 기업 정보, 채용 관련 일반 질의 응답
  - 개인 맞춤 분석이 필요한 경우: "자기소개서나 이력서를 첨부해 주세요" 안내
- [분석 모드]: 사용자가 파일을 첨부하거나 "자기소개서입니다", "이력서입니다" 등의 표현을 쓴 경우
  - 키워드 추출 → 사용자 성향 파악 → 선호도 질문 → 유사 기업 추천

2. 사용자 선호도 조사
분석 모드에서 다음 질문을 출력하고 반드시 사용자 응답을 기다린다:
- 관심 산업/분야는?
- 선호하는 면접 방식이나 고려할 사항은?
- 선호 지역은?
- 기업 규모 선호는?

3. 기업 정보 응답 형식 (API 또는 더미 데이터 기반):
- 기업명: [기업 이름]
- 산업 분야: [업종]
- 최근 채용공고: [요약 또는 링크]
- 근무 지역: [도시명]
- 주요 조건: [고용형태, 경력 등]

4. 시스템 작동 문구 금지
- “분석모드입니다”, “모드를 전환합니다” 등 표현은 사용하지 않는다.
- 분석 실패 시에는 “현재 정보만으로는 분석이 어렵습니다” 등 자연스러운 안내를 사용한다.

5. 검색 및 응답 처리
- 사용자가 “○○ 채용공고 알려줘”라고 입력하면
  - “○○ 채용공고 정보를 확인 중입니다...” 출력
  - (가능 시 API 결과 또는 GPT 요약 제공)

6. 컨텍스트 유지
- 사용자가 분석 도중 “다시 시작할래요”, “처음부터”라고 입력하면 상태 초기화
- 그 외에는 컨텍스트를 유지하며 대화 흐름을 이어간다

7. 대화 톤
- 친절하고 전문적인 어조
- 답변 가능한 범위 명확히 안내
- 불명확한 질문엔 구체화 요청
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("message", "")
        user_interest = request.form.get("interest", "")
        user_region = request.form.get("region", "")
        user_salary = request.form.get("salary", "")

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        preferences = extract_user_preferences(user_input) + [user_interest, user_region, user_salary]

        companies = build_company_list_from_job_api("개발")
        filtered = filter_companies(companies, user_interest, user_region, user_salary)

        if not filtered:
            reply = "현재 정보만으로는 조건에 맞는 기업을 찾기 어렵습니다. 입력하신 내용을 조금 바꾸어 다시 시도해 주세요."
            return jsonify({"reply": reply})

        prompt = build_recommendation_prompt(keywords, preferences, filtered)
        reply = get_gpt_reply(prompt)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

def extract_resume_text(text):
    return text

def extract_user_preferences(text):
    return re.findall(r"\d+\.\s*([^\n]*)", text)

def extract_keywords(text):
    prompt = f"다음 자기소개서에서 핵심 기술, 직무, 경험 키워드를 쉼표로 추출해줘:\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return [kw.strip() for kw in response.choices[0].message.content.split(",")]
    except Exception as e:
        print("❌ 키워드 추출 실패:", e)
        return []

@lru_cache(maxsize=100)
def build_company_list_from_job_api(keyword, rows=20):
    url = "https://118.67.151.173/data/api/jopblancApi.do"
    params = {
        "authKey": job_api_key,
        "callTp": "L",
        "listCount": rows,
        "query": keyword
    }
    try:
        response = requests.get(url, params=params, verify=False, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            companies = []
            for item in root.findall(".//jobList"):
                name = item.findtext("entrprsNm", "기업명 없음")
                tags = [
                    item.findtext("areaStr", ""),
                    item.findtext("emplymStleSeStr", ""),
                    item.findtext("dtyStr", ""),
                    item.findtext("pblancSj", "")
                ]
                tags = [t for t in " ".join(tags).split() if t]
                companies.append({"name": name, "tags": tags})
            return companies
    except Exception as e:
        print("❌ API 오류:", e)

    return [{"name": "더미기업", "tags": ["개발", "진주", "기술"]}]

def filter_companies(companies, interest, region, salary):
    filtered = []
    for c in companies:
        combined_tags = " ".join(c["tags"])
        if interest and interest not in combined_tags:
            continue
        if region and region not in combined_tags:
            continue
        filtered.append(c)
    return filtered[:3] if filtered else companies[:3]  # 조건 없으면 더미 3개라도 리턴

def compute_similarity(text1, text2):
    try:
        emb1 = client.embeddings.create(input=text1, model="text-embedding-ada-002").data[0].embedding
        emb2 = client.embeddings.create(input=text2, model="text-embedding-ada-002").data[0].embedding
        dot = sum(x * y for x, y in zip(emb1, emb2))
        norm1 = sum(x * x for x in emb1) ** 0.5
        norm2 = sum(y * y for y in emb2) ** 0.5
        return dot / (norm1 * norm2)
    except:
        return 0.0

def build_recommendation_prompt(keywords, preferences, companies):
    company_str = "\n".join([f"- {c['name']} ({', '.join(c['tags'])})" for c in companies])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join([p for p in preferences if p])}\n\n"
        f"[추천 기업 리스트]\n{company_str}\n\n"
        f"각 기업이 사용자에게 왜 적합한지 GPT 입장에서 설명해줘. 자기소개서처럼 말하지 마." 
    )

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT 응답 오류: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
