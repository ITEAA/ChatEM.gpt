from flask import Flask, request, jsonify, render_template, session
import time
import requests
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
assistant_id = os.getenv("ASSISTANT_ID")
job_api_key = os.getenv("JOB_API_KEY")


def extract_keywords_from_resume(text):
    prompt = f"다음 자기소개서에서 핵심 기술, 직무, 경험 키워드를 쉼표로 추출해줘:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return [kw.strip() for kw in response.choices[0].message.content.split(",")]


def parse_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]


def build_company_list_from_job_api(keyword, rows=50):
    url = "https://job.kosmes.or.kr/openApi/interestedJob/openApiJopblancList.do"
    params = {
        "serviceKey": job_api_key,
        "searchKeyword": keyword,
        "numOfRows": rows,
        "pageNo": 1,
        "returnType": "json"
    }
    response = requests.get(url, params=params)
    print("📡 API 요청 URL:", response.url)
    print("🔍 응답 상태 코드:", response.status_code)
    print("📦 응답 내용:", response.text[:1000])
    companies = []
    if response.status_code == 200:
        postings = response.json().get("data", [])
        for post in postings:
            tags = [post.get("rcritFieldNm", ""), post.get("regionNm", ""), post.get("emplmntTypeNm", "")]
            company = {
                "name": post.get("entrprsNm", "기업명 없음"),
                "tags": [t.strip() for t in tags if t.strip()]
            }
            companies.append(company)
    return companies


def match_company_to_user(companies, user_keywords, user_prefs):
    best = None
    best_score = -1
    for company in companies:
        overlap = set(user_keywords + user_prefs) & set(company["tags"])
        score = len(overlap)
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


def get_job_postings(keyword, rows=3):
    url = "https://job.kosmes.or.kr/openApi/interestedJob/openApiJopblancList.do"
    params = {
        "serviceKey": job_api_key,
        "searchKeyword": keyword,
        "numOfRows": rows,
        "pageNo": 1,
        "returnType": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []


@app.route("/")
def index():
    session.clear()
    session["stage"] = "awaiting_resume"
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "").strip()
        stage = session.get("stage", "awaiting_resume")

        if stage == "awaiting_resume":
            session["resume"] = user_message
            session["stage"] = "awaiting_preference"
            return jsonify(reply=(
                "자기소개서를 잘 받았습니다 😊\n"
                "이제 선호도를 알려주세요:\n"
                "1. 어떤 산업이나 분야에 관심이 있으신가요?\n"
                "2. 선호하는 면접 방식이나 고려사항이 있으신가요?\n"
                "3. 선호 지역은 어디인가요? (예: 서울, 부산, 재택)\n"
                "4. 기업 규모에 대한 선호가 있으신가요? (예: 스타트업, 대기업)"
            ))

        elif stage == "awaiting_preference":
            resume_text = session.get("resume", "")
            user_keywords = extract_keywords_from_resume(resume_text)
            user_preferences = parse_user_preferences(user_message)

            companies = []
            for keyword in user_keywords:
                companies = build_company_list_from_job_api(keyword)
                if companies:
                    break
            if not companies:
                companies = build_company_list_from_job_api(keyword="")

            matched_company = match_company_to_user(companies, user_keywords, user_preferences)

            if not matched_company:
                return jsonify(reply="죄송합니다. 추천 가능한 기업이 없습니다.")

            # 안전하게 tags 사용
            search_tag = matched_company["tags"][0] if matched_company.get("tags") and matched_company["tags"] else user_keywords[0]
            job_postings = get_job_postings(search_tag)

            job_summary = ""
            for job in job_postings:
                job_summary += f"- {job['entrprsNm']} | {job['title']} | {job['regionNm']} | {job['emplmntTypeNm']}\n링크: {job['linkUrl']}\n\n"

            prompt = build_explanation_prompt(user_keywords, user_preferences, matched_company, job_summary)

            thread = client.beta.threads.create()
            client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

            timeout = 30
            start_time = time.time()
            while run.status not in ["completed", "failed", "cancelled"]:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if time.time() - start_time > timeout:
                    return jsonify(reply="GPT 응답이 지연되고 있어요. 잠시 후 다시 시도해주세요.")

            messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
            session["stage"] = "complete"

            for msg in messages.data:
                for content in msg.content:
                    if content.type == "text":
                        return jsonify(reply=content.text.value)

            return jsonify(reply="GPT로부터 응답을 받지 못했습니다.")

        else:
            return jsonify(reply="처음부터 다시 시작하시려면 '처음부터'라고 입력해주세요.")

    except Exception as e:
        print("❌ 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
