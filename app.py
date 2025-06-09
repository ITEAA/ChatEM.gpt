from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import json
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, send_from_directory
import os

app = Flask(__name__)

# ✅ OpenAI API 키 설정
openai.api_key = "your-openai-api-key"

# ✅ ChatEM 기업 top20 데이터 로드
with open("ChatEM_top20_companies.json", encoding="utf-8") as f:
    top20_data = json.load(f)
df_companies = pd.DataFrame(top20_data)

# ✅ GPT 키워드 추출 함수
def extract_keywords_gpt(text):
    prompt = f"다음 자기소개서에서 핵심 키워드 5개를 뽑아줘:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip().split('\n')

# ✅ 기업 추천 함수
def recommend_companies(user_keywords):
    user_profile = " ".join(user_keywords)
    df_companies["combined_text"] = (
        df_companies["name"].fillna('') + " " +
        df_companies["summary"].fillna('') + " " +
        df_companies["region"].fillna('') + " " +
        df_companies["industry"].fillna('')
    )
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_companies["combined_text"].tolist() + [user_profile])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    df_companies["유사도"] = cosine_sim
    top = df_companies.sort_values(by="유사도", ascending=False).head(3)
    return top

# ✅ index.html 렌더링
@app.route("/")
def index():
    return render_template("index.html")

# ✅ 추천 결과 JSON 응답
@app.route("/recommend", methods=["POST"])
def recommend():
    user_data = request.json
    text = user_data.get("text", "")
    region = user_data.get("region", "")
    salary = user_data.get("salary", "")
    interest = user_data.get("interest", "")

    keywords = extract_keywords_gpt(text)
    if region:
        keywords.append(region)
    if salary:
        keywords.append(salary)
    if interest:
        keywords.append(interest)

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

if __name__ == "__main__":
    app.run(debug=True)
