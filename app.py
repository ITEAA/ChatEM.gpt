from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import openai
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ✅ OpenAI API 키 설정
openai.api_key = "your-openai-api-key"  # 실제 키로 교체

# ✅ ChatEM 기업 데이터 로드
with open("ChatEM_top20_companies.json", encoding="utf-8") as f:
    top20_data = json.load(f)
df_companies = pd.DataFrame(top20_data)

# ✅ PDF 또는 텍스트 파일 읽기
def read_file_content(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# ✅ GPT 키워드 추출
def extract_keywords_gpt(text):
    prompt = f"다음 자기소개서에서 핵심 키워드 5개를 뽑아줘:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip().split('\n')

# ✅ 기업 추천
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

# ✅ index 페이지
@app.route("/")
def index():
    return render_template("index.html")

# ✅ 파일 또는 텍스트 기반 추천 라우트
@app.route("/chat", methods=["POST"])
def chat():
    file = request.files.get("file")
    text = request.form.get("message", "").strip()
    interest = request.form.get("interest", "").strip()
    region = request.form.get("region", "").strip()
    salary = request.form.get("salary", "").strip()

    # 🔍 파일이 있으면 파일 내용 사용
    if file:
        try:
            text = read_file_content(file)
        except Exception as e:
            return jsonify({"reply": f"파일을 읽는 중 오류 발생: {str(e)}"}), 400

    if not text:
        return jsonify({"reply": "자기소개서 내용을 입력하거나 파일을 업로드해주세요."}), 400

    # 🔍 키워드 추출
    try:
        keywords = extract_keywords_gpt(text)
        if interest: keywords.append(interest)
        if region: keywords.append(region)
        if salary: keywords.append(salary)
    except Exception as e:
        return jsonify({"reply": f"GPT 키워드 추출 실패: {str(e)}"}), 500

    # 🔍 기업 추천
    try:
        results = recommend_companies(keywords)
        descriptions = []
        for _, row in results.iterrows():
            desc = f"{row['name']}에서 {row['summary']} 직무를 채용 중이며, 위치는 {row['region']}입니다."
            descriptions.append(desc)
        reply = "\n\n".join(descriptions)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"기업 추천 중 오류 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
