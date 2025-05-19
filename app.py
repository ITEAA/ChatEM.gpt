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
app.secret_key = "your-secret-key"

assistant_id = os.getenv("ASSISTANT_ID")
job_api_key = "fYL5gLDcPZ/iE6TB7Rmg1AnxisbHUUFMUuK8Am/MxcIC5+G2awO4kGH6CjFbgwAorXjRlhuqogcHGSEyLzQXdoOW2XonGbNFkASwL8QBm6FkiXgC/hHz+Jr/HAInzOPG"

def extract_keywords_from_resume(text):
    prompt = f"""
    다음 이력서에서 핵심 기술, 직무, 경험 키워드만 쉼표로 구분하여 추출해줘.
    결과는 키워드 리스트로만 짧게 출력해줘.

    이력서 내용:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return [kw.strip() for kw in response.choices[0].message.content.split(",")]

def parse_user_preferences(text):
    prefs = re.findall(r"\d+\.\s*([^\n]*)", text)
    return [p.strip() for p in prefs]

def build_company_list_from_job_api(keyword="AI", rows=50):
    url = "https://job.kosmes.or.kr/openApi/interestedJob/openApiJopblancList.do"
    params = {
        "serviceKey": job_api_key,
        "searchKeyword": keyword,
        "numOfRows": rows,
        "pageNo": 1,
        "returnType": "json"
    }
    response = requests.get(url, params=params)
    companies = []
    if response.status_code == 200:
        postings = response.json().get("data", [])
        for post in postings:
            tags = [
                post.get("rcritFieldNm", ""),
                post.get("regionNm", ""),
                post.get("emplmntTypeNm", "")
            ]
            company = {
                "name": post.get("entrprsNm", "기업명 없음"),
                "tags": [tag.strip() for tag in tags if tag.strip()]
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
    return best

def build_explanation_prompt(keywords, preferences, company, job_summary=""):
    base = f"""
    다음 사용자 정보와 추천 기업을 기반으로, 왜 이 기업이 적합한지 2~3문장으로 설명해주세요.

    [사용자 정보]
    - 기술 키워드: {', '.join(keywords)}
    - 선호: {', '.join(preferences)}

    [추천 기업]
    - 기업명: {company['name']}
    - 태그: {', '.join(company['tags'])}
    """
    if job_summary:
        base += f"\n\n[관련 채용공고]\n{job_summary}"
    return base

def get_job_postings(keyword="AI", page=1, rows=3):
    url = "https://job.kosmes.or.kr/openApi/interestedJob/openApiJopblancList.do"
    params = {
        "serviceKey": job_api_key,
        "searchKeyword": keyword,
        "numOfRows": rows,
        "pageNo": page,
        "returnType": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form.get("message", "").strip()
        uploaded_file = request.files.get("file")

        if not uploaded_file or not user_message:
            return jsonify(reply="자기소개서와 선호도 입력이 모두 필요합니다.")

        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        user_keywords = extract_keywords_from_resume(file_content)
        user_preferences = parse_user_preferences(user_message)

        companies = build_company_list_from_job_api(keyword=user_keywords[0] if user_keywords else "AI")
        matched_company = match_company_to_user(companies, user_keywords, user_preferences)

        job_postings = get_job_postings(matched_company["tags"][0])
        job_summary = ""
        for job in job_postings:
            job_summary += f"- {job['entrprsNm']} | {job['title']} | {job['regionNm']} | {job['emplmntTypeNm']}\n링크: {job['linkUrl']}\n\n"

        prompt = build_explanation_prompt(user_keywords, user_preferences, matched_company, job_summary)

        if "thread_id" not in session:
            thread = client.beta.threads.create()
            session["thread_id"] = thread.id
        else:
            thread = client.beta.threads.retrieve(session["thread_id"])

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        timeout = 30
        start_time = time.time()
        while run.status not in ["completed", "failed", "cancelled"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if time.time() - start_time > timeout:
                return jsonify(reply="GPT 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요.")

        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
        for msg in messages.data:
            for content in msg.content:
                if content.type == "text":
                    return jsonify(reply=content.text.value)

        return jsonify(reply="GPT로부터 응답을 받지 못했습니다.")

    except Exception as e:
        print("❌ 서버 에러:", str(e))
        return jsonify(reply="서버 오류 발생: " + str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
