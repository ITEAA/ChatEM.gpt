import os
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ✅ OPENAI 키 불러오기 (.env 대신 fly.io secrets 사용)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 더미 기업 데이터 로드
with open("ChatEM_top20_companies.json", "r", encoding="utf-8") as f:
    companies = json.load(f)

# TF-IDF 기반 유사도 계산 함수
def compute_similarity(user_text, companies):
    corpus = [user_text] + [c["description"] for c in companies]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    user_text = data.get("text")

    if not user_text:
        return jsonify({"error": "자기소개서 내용이 없습니다."}), 400

    # ✅ GPT로 키워드 추출 (ChatCompletion 최신 방식 사용)
    prompt = f"""
다음 자기소개서에서 핵심 키워드를 5개 추출해줘. 
형식: 키워드1, 키워드2, ...
자기소개서:
{user_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 채용담당자야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        keywords = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"GPT 응답 오류: {str(e)}"}), 500

    # TF-IDF 유사도 계산
    similarity_scores = compute_similarity(user_text, companies)
    top_indices = similarity_scores.argsort()[-3:][::-1]
    top_matches = [
        {
            "company": companies[i]["name"],
            "score": round(similarity_scores[i] * 100, 2),
            "description": companies[i]["description"]
        }
        for i in top_indices
    ]

    return jsonify({
        "keywords": keywords,
        "matches": top_matches
    })

if __name__ == "__main__":
    app.run(debug=True)
