from flask import Flask, request, jsonify, render_template
import os, re, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Load Jinju companies JSON
with open("jinju_companies.json", "r", encoding="utf-8") as f:
    COMPANY_DB = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("message", "")
        interest = request.form.get("interest", "")
        region = request.form.get("region", "")
        salary = request.form.get("salary", "")

        resume = extract_resume_text(user_input)
        keywords = extract_keywords(resume)
        prefs = [interest, region, salary]

        # 필터링 및 매칭
        matches = match_companies(keywords, prefs)
        if not matches:
            return jsonify({"reply": "조건에 맞는 기업을 찾지 못했습니다. 입력값을 조정해 보세요."})

        prompt = build_prompt(keywords, prefs, matches)
        reply = get_gpt_reply(prompt)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

def extract_resume_text(text):
    return text

def extract_keywords(text):
    prompt = f"다음 자기소개서에서 핵심 키워드(기술, 경험, 직무)를 쉼표로 추출해줘:\n{text}"
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

def match_companies(keywords, preferences):
    interest, region, salary = preferences
    scored = []

    for c in COMPANY_DB:
        score = 0
        content = f"{c['name']} {c['industry']} {c['region']} {c['summary']}"
        for kw in keywords:
            if kw in content:
                score += 1
        if interest and interest in c['industry']:
            score += 2
        if region and region in c['region']:
            score += 2
        scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored[:3] if s > 0]

def build_prompt(keywords, prefs, companies):
    intro = (
        "너는 지금부터 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 추천해주는 역할을 수행한다. 다음 지침을 따르라:\n"
        "- 사용자 키워드: " + ", ".join(keywords) + "\n"
        "- 사용자 선호: " + ", ".join([p for p in prefs if p]) + "\n"
        "- 추천 기업:\n"
    )
    for c in companies:
        intro += f"- 기업명: {c['name']}, 업종: {c['industry']}, 지역: {c['region']}, 요약: {c['summary']}\n"
    intro += "이 기업들이 왜 사용자의 성향과 잘 맞는지 설명해줘."
    return intro

def get_gpt_reply(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT 오류: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
