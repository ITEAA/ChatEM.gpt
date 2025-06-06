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
        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        user_prefs = extract_user_preferences(user_input)

        companies = build_company_list_from_job_api("개발")
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
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            companies = []
            for item in root.findall(".//jobList"):
                name = item.findtext("entrprsNm", "기업명 없음")
                tags = [
                    item.findtext("areaStr", ""),
                    item.findtext("emplymStleSeStr", ""),
                    item.findtext("dtyStr", "")
                ]
                tags += item.findtext("pblancSj", "").split()
                companies.append({"name": name, "tags": [t for t in tags if t]})
            return companies
    except Exception as e:
        print("❌ API 오류:", e)

    return [{"name": "더미기업", "tags": ["개발", "진주", "기술"]}]

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

def match_company_to_user(companies, user_keywords, user_prefs):
    user_text = " ".join(user_keywords + user_prefs)
    best, best_score = None, -1
    for company in companies:
        score = compute_similarity(user_text, " ".join(company["tags"]))
        if score > best_score:
            best, best_score = company, score
    return best

def build_explanation_prompt(keywords, preferences, company):
    return (
        f"다음 사용자 정보와 추천 기업을 기반으로, 왜 이 기업이 적합한지 설명해주세요.\n\n"
        f"[사용자 정보]\n- 키워드: {', '.join(keywords)}\n- 선호: {', '.join(preferences)}\n\n"
        f"[추천 기업]\n- 기업명: {company['name']}\n- 태그: {', '.join(company['tags'])}"
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
