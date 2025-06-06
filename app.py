from flask import Flask, request, jsonify, render_template
import os, re, json, requests, xml.etree.ElementTree as ET
from dotenv import load_dotenv
from functools import lru_cache
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
client = OpenAI()

app.secret_key = os.getenv("FLASK_SECRET_KEY")
job_api_key = os.getenv("JOB_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("message", "")
        user_interest = request.form.get("interest", "").strip()
        user_region = request.form.get("region", "").strip()
        user_salary = request.form.get("salary", "").strip()

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        preferences = extract_user_preferences(user_input) + [user_interest, user_region, user_salary]
        preferences = [p for p in preferences if p]  # 빈 문자열 제거

        companies = build_company_list_from_job_api("개발") or load_dummy_companies()
        matched = match_top_companies(companies, keywords + preferences)

        if not matched:
            return jsonify({"reply": "❌ 조건에 맞는 기업 정보를 찾지 못했습니다. 입력 조건을 줄여 다시 시도해보세요."})

        prompt = build_explanation_prompt(keywords, preferences, matched)
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
        return [kw.strip() for kw in response.choices[0].message.content.split(",") if kw.strip()]
    except Exception as e:
        print("❌ 키워드 추출 실패:", e)
        return []

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
    return None

def load_dummy_companies():
    try:
        with open("dummy_companies.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("❌ 더미 데이터 불러오기 실패:", e)
        return [{"name": "더미기업", "tags": ["개발", "진주", "기술"]}]

def match_top_companies(companies, user_profile, top_n=3):
    results = []
    user_text = " ".join(user_profile)
    for company in companies:
        comp_text = " ".join(company.get("tags", []))
        score = compute_similarity(user_text, comp_text)
        results.append((company, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_n] if r[1] > 0.1]

def compute_similarity(text1, text2):
    try:
        emb1 = client.embeddings.create(input=text1, model="text-embedding-ada-002").data[0].embedding
        emb2 = client.embeddings.create(input=text2, model="text-embedding-ada-002").data[0].embedding
        dot = sum(x * y for x, y in zip(emb1, emb2))
        norm1 = sum(x * x for x in emb1) ** 0.5
        norm2 = sum(y * y for y in emb2) ** 0.5
        return dot / (norm1 * norm2)
    except Exception as e:
        print("❌ 임베딩 오류:", e)
        return 0.0

def build_explanation_prompt(keywords, preferences, companies):
    company_list = "\n".join([f"- {c['name']} ({', '.join(c['tags'])})" for c in companies])
    return (
        "너는 지금부터 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 추천해주는 역할을 수행한다. 다음 지침을 따르라:\n\n"
        "1. 분석 실패 시에는 '현재 정보만으로는 분석이 어렵습니다' 등 자연스럽게 안내하고 시스템 문구는 금지한다.\n"
        "2. 아래 사용자 정보와 기업 리스트를 참고해 추천 이유를 설명한다.\n\n"
        f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join(preferences)}\n\n"
        f"[추천 기업 후보]\n{company_list}\n\n"
        "각 기업이 왜 사용자에게 적합한지 분석해서 자연스럽게 설명해줘."
    )

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT 응답 오류: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
