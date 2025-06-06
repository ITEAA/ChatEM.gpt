from flask import Flask, request, jsonify, render_template
import os, json, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

# Load local JSON companies file
with open("jinju_companies.json", encoding="utf-8") as f:
    LOCAL_COMPANIES = json.load(f)

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

        resume_text = extract_resume_text(user_input)
        keywords = extract_keywords(resume_text)

        preferences = [user_interest, user_region, user_salary]
        companies = filter_local_companies(keywords, preferences)

        if not companies:
            return jsonify({"reply": "조건에 맞는 기업을 찾지 못했습니다. 입력값을 조정해 보세요."})

        prompt = build_prompt(keywords, preferences, companies)
        reply = get_gpt_reply(prompt)

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"❌ 서버 오류: {str(e)}"}), 500

def extract_resume_text(text):
    return text

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

def filter_local_companies(keywords, preferences):
    interest, region, _ = preferences
    matches = []

    for company in LOCAL_COMPANIES:
        tags = (company.get("산업", "") + " " + company.get("소개", "")).lower()
        name = company.get("기업명", "")

        # 필터 조건 중 하나라도 만족하면 추가
        if any(k.lower() in tags for k in keywords) or any(p.lower() in tags for p in preferences if p):
            matches.append(company)

    return matches[:3] if matches else LOCAL_COMPANIES[:3]

def build_prompt(keywords, preferences, companies):
    company_list = "\n".join([
        f"- 기업명: {c['기업명']}\n  산업 분야: {c['산업']}\n  지역: {c['지역']}\n  소개: {c['소개']}"
        for c in companies
    ])
    return (
        f"너는 지금부터 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 추천해주는 역할을 수행한다. 다음 지침을 따르라:\n"
        f"\n[사용자 정보]\n키워드: {', '.join(keywords)}\n선호: {', '.join([p for p in preferences if p])}\n"
        f"\n[기업 리스트]\n{company_list}\n"
        f"\n위 기업들 중 사용자의 성향과 가장 잘 맞는 회사를 1~2개 추천하고 이유를 설명해줘."
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
