from flask import Flask, request, jsonify, render_template
import os
import re
import requests
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
proxy_url = os.getenv("PROXY_URL", "http://127.0.0.1:5001/proxy")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("message", "")

        if len(user_input.strip()) < 5:
            return jsonify({"reply": "간단한 인사말보다는 관심 분야나 자기소개서 내용을 입력해 주세요 😊"})

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        user_prefs = extract_user_preferences(user_input)

        companies = build_company_list_from_proxy_api("개발")
        match = match_company_to_user(companies, keywords, user_prefs)

        if not match:
            return jsonify({"reply": "❌ 기업 정보를 불러오지 못했습니다. 나중에 다시 시도해주세요."})

        prompt = build_explanation_prompt(keywords, user_prefs, match)
        reply = get_gpt_reply(prompt)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

def extract_resume_text(text):
    return text

def extract_user_preferences(text):
    return parse_user_preferences(text)

def extract_keywords(text):
    prompt = f"다음 자기소개서에서 핵심 키워드를 쉼표로 추출해줘:\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return [kw.strip() for kw in response.choices[0].message.content.split(",") if kw.strip()]
    except Exception as e:
        print("❌ 키워드 추출 실패:", e)
        return ["AI", "프로그래밍", "팀워크"]

def parse_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]

@lru_cache(maxsize=100)
def build_company_list_from_proxy_api(keyword, rows=10):
    try:
        params = {
            "authKey": job_api_key,
            "callTp": "L",
            "listCount": rows,
            "query": keyword,
        }
        response = requests.get(proxy_url, params=params, timeout=10)
        companies = []
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall(".//jobList"):
                name = item.findtext("entrprsNm", "기업명 없음")
                tags = [item.findtext(k, '') for k in ["areaStr", "emplymStleSeStr", "dtyStr"] if item.findtext(k)]
                title = item.findtext("pblancSj", "")
                tags += title.split()
                companies.append({"name": name, "tags": tags})
            return companies
    except Exception as e:
        print("❌ Proxy API 오류:", str(e))
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
    best, best_score = None, -1
    for company in companies:
        company_text = " ".join(company["tags"])
        score = compute_similarity(user_text, company_text)
        if score > best_score:
            best, best_score = company, score
    return best

def build_explanation_prompt(keywords, preferences, company):
    base = f"사용자 정보와 추천 기업 기반 설명:\n"
    base += f"[키워드] {', '.join(keywords)}\n[선호] {', '.join(preferences)}\n"
    base += f"[추천 기업] {company['name']}\n[태그] {', '.join(company['tags'])}"
    return base

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ GPT 응답 실패:", e)
        return "GPT 분석에 실패했습니다. 다시 시도해 주세요."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
