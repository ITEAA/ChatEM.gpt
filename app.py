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
            return jsonify({"reply": "❌ 조건에 맞는 기업 정보를 찾지 못했습니다. 더 다양한 키워드나 조건으로 다시 시도해 주세요."})

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
    return filtered[:3]

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
        f"[사용자 정보]\n키워드: {', '.join(keywords)}\n선호: {', '.join([p for p in preferences if p])}\n\n"
        f"[추천 기업 리스트]\n{company_str}\n\n"
        f"이 사용자에게 위 기업들이 왜 적합한지 챗봇 시점에서 객관적으로 설명해줘. 사용자 입장이 아니라 분석자 입장에서 써줘."
    )

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT 응답 오류: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
