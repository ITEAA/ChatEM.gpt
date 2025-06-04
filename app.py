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


# 🔸 index.html 보여주기
@app.route("/")
def home():
    return render_template("index.html")


# 🔸 파일 또는 텍스트 메시지를 POST로 받아 GPT 응답 처리
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "")
    file = request.files.get("file", None)

    # 텍스트 또는 파일 기반 키워드 추출
    input_text = ""
    if file:
        input_text = file.read().decode("utf-8")
    elif user_message:
        input_text = user_message

    if not input_text:
        return jsonify({"reply": "❌ 빈 메시지 또는 파일입니다."})

    keywords = extract_keywords_from_resume(input_text)
    user_prefs = ["창의적인 분위기", "개발 직무", "유연근무제 선호"]  # 예시 고정값

    companies = build_company_list_from_job_api(keywords[0] if keywords else "개발", rows=10)
    match = match_company_to_user(companies, keywords, user_prefs)

    prompt = build_explanation_prompt(keywords, user_prefs, match)
    try:
        gpt_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        reply = gpt_response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"❌ 추천 설명 생성 실패: {e}"

    return jsonify({"reply": reply})


# 🔸 키워드 추출
def extract_keywords_from_resume(text):
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


def parse_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]


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
        return companies

    except Exception as e:
        print("❌ API 요청 오류:", str(e))
        return []


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


def build_explanation_prompt(keywords, preferences, company, job_summary=""):
    base = f"다음 사용자 정보와 추천 기업을 기반으로, 왜 이 기업이 적합한지 설명해주세요.\n\n"
    base += f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join(preferences)}\n\n"
    base += f"[추천 기업]\n- 기업명: {company['name']}\n- 태그: {', '.join(company['tags'])}"
    if job_summary:
        base += f"\n\n[채용공고]\n{job_summary}"
    return base


# 🔸 Fly.io 호환 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
