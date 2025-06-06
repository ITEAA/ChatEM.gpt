from flask import Flask, request, jsonify, render_template
import os, re, json
from dotenv import load_dotenv
from functools import lru_cache
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

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

        companies = load_companies_from_json()
        filtered = filter_companies(companies, user_interest, user_region)

        if not filtered:
            filtered = companies[:3]  # fallback

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

@lru_cache(maxsize=1)
def load_companies_from_json():
    with open("jinju_companies.json", "r", encoding="utf-8") as f:
        return json.load(f)

def filter_companies(companies, interest, region):
    filtered = []
    for c in companies:
        tags = " ".join(c.get("tags", []))
        if interest and interest not in tags:
            continue
        if region and region not in tags:
            continue
        filtered.append(c)
    return filtered[:3]

def build_recommendation_prompt(keywords, preferences, companies):
    company_str = "\n".join([f"- {c['name']} ({', '.join(c['tags'])})" for c in companies])
    return (
        "다음은 한 사용자의 자기소개서 키워드와 선호도, 그리고 추천 후보 기업입니다."
        "\n이 사용자의 특성과 선호도에 가장 적합한 기업 1~2곳을 골라 추천해 주세요."
        "\n설명은 따뜻하고 전문적인 어조로 작성해 주세요."
        f"\n\n[사용자 키워드]\n{', '.join(keywords)}"
        f"\n[사용자 선호도]\n{', '.join([p for p in preferences if p])}"
        f"\n\n[기업 후보]\n{company_str}"
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
