from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = Flask(__name__, template_folder="templates")
CORS(app)

GG_API_KEY = "8af0f404ca144249be0cfab9728b619b"
GG_API_URL = "https://openapi.gg.go.kr/EmplmntInfoStus"
openai.api_key = "your-openai-key"  # 실제 키로 교체하세요

@app.route("/")
def index():
    return render_template("index_local.html")

def fetch_employment_info(index=1, size=100):
    params = {"KEY": GG_API_KEY, "Type": "xml", "pIndex": index, "pSize": size}
    response = requests.get(GG_API_URL, params=params)
    root = ET.fromstring(response.content)
    rows = root.findall(".//row")
    data = []
    for row in rows:
        row_data = [row.find(col).text if row.find(col) is not None else "" for col in [
            "REGIST_DE", "SIGUN_NM", "COMPNY_NM", "EMPLMNT_TITLE", "WAGE_FORM", "SALARY_INFO", "WORK_REGION_LOC",
            "WORK_FORM", "MIN_ACDMCR", "CAREER_INFO", "CLOS_DE_INFO", "EMPLMNT_INFO_URL"
        ]]
        data.append(row_data)
    columns = ["등록일자", "시군명", "회사명", "채용공고명", "임금형태", "급여", "근무지역", "근무형태", "최소학력", "경력", "마감일자", "채용정보URL"]
    return pd.DataFrame(data, columns=columns)

def extract_keywords_gpt(text):
    prompt = f"다음 자기소개서에서 핵심 키워드 5개를 뽑아줘:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip().split('\n')

def recommend_jobs(df, keywords, filters):
    if filters.get("region"):
        df = df[df["근무지역"].str.contains(filters["region"])]
    if filters.get("career"):
        df = df[df["경력"].str.contains(filters["career"])]
    corpus = df["채용공고명"].fillna("").tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [" ".join(keywords)])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df["유사도"] = cosine_sim[0]
    top = df.sort_values(by="유사도", ascending=False).head(3)
    return top[["회사명", "채용공고명", "근무지역", "급여", "채용정보URL", "유사도"]]

@app.route("/recommend-local", methods=["POST"])
def recommend_local():
    user_data = request.json
    text = user_data.get("text", "")
    filters = {
        "region": user_data.get("region"),
        "career": user_data.get("career")
    }
    keywords = extract_keywords_gpt(text)
    df_jobs = fetch_employment_info()
    recommendations = recommend_jobs(df_jobs, keywords, filters)
    results = []
    for _, row in recommendations.iterrows():
        desc = f"{row['회사명']}에서 {row['채용공고명']} 포지션을 모집 중이며, 위치는 {row['근무지역']}, 급여는 {row['급여']}입니다."
        results.append({
            "company": row["회사명"],
            "position": row["채용공고명"],
            "location": row["근무지역"],
            "salary": row["급여"],
            "url": row["채용정보URL"],
            "score": round(float(row["유사도"]), 2),
            "description": desc
        })
    return jsonify(results)

@app.route("/chat-local", methods=["POST"])
def chat_local():
    user_message = request.json.get("message", "")
    if "진주" in user_message:
        df_jobs = fetch_employment_info()
        df_filtered = df_jobs[df_jobs["근무지역"].str.contains("진주")]
        return jsonify(df_filtered.head(5).to_dict(orient="records"))
    return jsonify({"message": "다시 한 번 이력서를 입력해 주세요."})

if __name__ == "__main__":
    app.run(debug=True)
