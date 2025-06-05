from flask import Flask, request, jsonify, render_template
import time
import requests
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
assistant_id = os.getenv("ASSISTANT_ID")
job_api_key = os.getenv("JOB_API_KEY")

# 🔸 HTML 렌더링
@app.route("/")
def home():
    return render_template("index.html")

# 🔸 채팅 처리
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("message", "")
        # 파일도 필요하면 file = request.files.get("file")

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        user_prefs = extract_user_preferences(user_input)

        dummy_companies = [
            {"name": "경남IT솔루션", "tags": ["진주", "소프트웨어", "개발", "백엔드"]},
            {"name": "진주로직스", "tags": ["물류", "운송", "경상남도", "물류관리"]},
            {"name": "에코그린테크", "tags": ["환경", "에너지", "친환경", "진주"]},
            {"name": "네오교육", "tags": ["에듀테크", "교육", "콘텐츠", "웹"]},
        ]

        match = match_company_to_user(dummy_companies, keywords, user_prefs)

        if not match:
            return jsonify({"reply": "❌ 기업 정보를 불러오지 못했습니다. 나중에 다시 시도해주세요."})

        prompt = build_explanation_prompt(keywords, user_prefs, match)
        reply = get_gpt_reply(prompt)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

# 🔸 간단한 이력서 추출기
def extract_resume_text(text):
    return text

# 🔸 사용자 선호 추출기
def extract_user_preferences(text):
    return parse_user_preferences(text)

# 🔸 키워드 추출
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
        return ["개발", "팀워크", "문제해결"]

# 🔸 사용자 선호 파싱
def parse_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]

# 🔸 API로 기업 리스트 생성
@lru_cache(maxsize=100)
def build_company_list_from_job_api(keyword, rows=10):
    url = "https://118.67.151.173/data/api/jopblancApi.do"
    params = {
        "authKey": job_api_key,
        "callTp": "L",
        "listCount": rows,
        "query": keyword
    }
    try:
        response = requests.get(url, params=params, verify=False, timeout=10)
        print("📡 API 요청 URL:", response.url)
        print("🔍 응답 상태 코드:", response.status_code)

        companies = []
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall(".//jobList"):
                name = item.findtext("entrprsNm", "기업명 없음")
                area = item.findtext("areaStr", "")
                style = item.findtext("emplymStleSeStr", "")
                duty = item.findtext("dtyStr", "")
                title = item.findtext("pblancSj", "")
                tags = [t for t in [area, style, duty] if t]
                tags += title.split()
                companies.append({"name": name, "tags": tags})
            if companies:
                return companies
    except Exception as e:
        print("❌ API 요청 오류:", str(e))

    #API 실패 시 더미 데이터 사용
    print("⚠️ API 실패. 더미 기업 리스트 사용.")
    return [
        {"name": "한국세라믹기술원", "tags": ["세라믹", "연구개발", "재료", "진주"]},
        {"name": "한국남동발전(주)", "tags": ["에너지", "발전소", "전기", "공기업", "진주"]},
        {"name": "(주)휴먼아이티솔루션", "tags": ["IT", "의료정보", "소프트웨어", "진주"]},
        {"name": "대호테크", "tags": ["자동차부품", "생산", "기계설비", "진주"]},
        {"name": "(주)지엠텍", "tags": ["드론", "정밀측량", "항공촬영", "진주", "ICT"]},
    ]

# 🔸 코사인 유사도 계산
def compute_similarity(text1, text2):
    try:
        emb1 = client.embeddings.create(input=text1, model="text-embedding-ada-002").data[0].embedding
        emb2 = client.embeddings.create(input=text2, model="text-embedding-ada-002").data[0].embedding
        return cosine_similarity(emb1, emb2)
    except Exception as e:
        print("❌ 유사도 계산 실패:", str(e))
        return 0.0

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)

# 🔸 기업 매칭
def match_company_to_user(companies, user_keywords, user_prefs):
    user_text = " ".join(user_keywords + user_prefs)
    best = None
    best_score = -1

    for company in companies:
        company_text = " ".join(company["tags"])
        score = compute_similarity(user_text, company_text)
        if score > best_score:
            best = company
            best_score = score
    return best or (companies[0] if companies else None)

# 🔸 GPT 설명용 프롬프트 생성
def build_explanation_prompt(keywords, preferences, company, job_summary=""):
    base = f"다음 사용자 정보와 추천 기업을 기반으로, 왜 이 기업이 적합한지 설명해주세요.\n\n"
    base += f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join(preferences)}\n\n"
    
    if company is None:
        base += "[추천 기업 정보 없음]\n- 기업 추천에 실패했습니다."
        return base

    base += f"[추천 기업]\n- 기업명: {company['name']}\n- 태그: {', '.join(company['tags'])}"
    if job_summary:
        base += f"\n\n[채용공고]\n{job_summary}"
    return base

# 🔸 GPT 응답
def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ GPT 응답 오류:", str(e))
        return "❌ GPT 응답에 실패했습니다. 나중에 다시 시도해주세요."

# 🔸 Fly.io 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
