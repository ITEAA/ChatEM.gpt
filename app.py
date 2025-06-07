
from flask import Flask, request, jsonify
import pandas as pd
import requests
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = Flask(__name__)

# ✅ GPT API 키 설정
openai.api_key = "your-openai-api-key"

# ✅ Top 20 기업 데이터 불러오기
with open("ChatEM_top20_companies.json", encoding="utf-8") as f:
    top20_companies = json.load(f)
df_companies = pd.DataFrame(top20_companies)

# ✅ GPT 기반 키워드 추출 함수
def extract_keywords_gpt(text):
    prompt = f"다음 자기소개서에서 핵심 키워드 5개를 뽑아줘:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip().split('\n')

# ✅ TF-IDF 기반 유사도 추천
def recommend_companies(user_keywords):
    user_profile = " ".join(user_keywords)
    df_companies["combined_text"] = df_companies["name"].fillna('') + " " +                                      df_companies["summary"].fillna('') + " " +                                      df_companies["region"].fillna('') + " " +                                      df_companies["industry"].fillna('')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_companies["combined_text"].tolist() + [user_profile])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    df_companies["유사도"] = cosine_sim
    top = df_companies.sort_values(by="유사도", ascending=False).head(3)
    return top

# ✅ 추천 라우트
@app.route("/recommend", methods=["POST"])
def recommend():
    user_data = request.json
    text = user_data.get("text", "")
    region = user_data.get("region", "")
    salary = user_data.get("salary", "")
    interest = user_data.get("interest", "")

    # 키워드 추출
    keywords = extract_keywords_gpt(text)
    if region: keywords.append(region)
    if salary: keywords.append(salary)
    if interest: keywords.append(interest)

    results = recommend_companies(keywords)

    output = []
    for _, row in results.iterrows():
        desc = f"{row['name']}에서 {row['summary']} 직무를 채용 중이며, 위치는 {row['region']}입니다."
        output.append({
            "company": row["name"],
            "location": row["region"],
            "industry": row["industry"],
            "url": row["url"],
            "score": round(float(row["유사도"]), 2),
            "description": desc
        })
    return jsonify(output)

# ✅ 홈 테스트용 라우트
@app.route("/")
def index():
    return "✅ ChatEM 로컬 서버 정상 작동 중"

if __name__ == "__main__":
    app.run(debug=True)
