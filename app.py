import os
import json
import requests
from flask import Flask, request, render_template
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_keywords(text):
    prompt = f"""
다음 자기소개서에서 핵심 키워드 5개를 추출해줘. 각 키워드는 1~3단어 이내로 하고, 쉼표로 구분해서 출력해줘.

{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        keywords = response.choices[0].message.content.strip()
        return [k.strip() for k in keywords.split(",") if k.strip()]
    except Exception as e:
        print(f"❌ 키워드 추출 오류: {e}")
        return []

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ 임베딩 생성 오류: {e}")
        return []

def cosine_similarity(vec1, vec2):
    try:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    except:
        return 0.0

def load_dummy_companies():
    try:
        with open("dummy_companies.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def get_companies(query):
    try:
        raise Exception("프록시 서버 비활성화로 API 생략")
    except Exception as e:
        print(f"❌ API 프록시 요청 실패: {e}")
        print("⚠️ API 실패. 더미 기업 리스트 사용.")
        return load_dummy_companies()

def get_gpt_reply(prompt):
    try:
        print("🧪 GPT 프롬프트 길이:", len(prompt))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT 응답 오류:", e)
        return "죄송합니다. 현재 추천을 제공할 수 없습니다."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("user_input", "")

        if not user_input.strip():
            return render_template("index.html", response="입력된 내용이 없습니다.")

        # 너무 짧은 인삿말은 분석 제외
        if len(user_input.replace("\n", "").replace(" ", "").strip()) < 10:
            return render_template("index.html", response="안녕하세요! 원하시는 직무나 관심 분야, 또는 자기소개서를 입력해 주시면 맞춤 기업을 추천해드릴게요.")

        # 1. 키워드 추출
        keywords = extract_keywords(user_input)
        keyword_str = ", ".join(keywords)

        # 2. 사용자 임베딩 생성
        user_embedding = get_embedding(user_input)

        # 3. 기업 정보 가져오기
        companies = get_companies(query=keywords[0] if keywords else "개발")

        # 4. 기업 임베딩 및 유사도 계산
        scored_companies = []
        for company in companies:
            description = company.get("description", "")
            company_embedding = get_embedding(description)
            score = cosine_similarity(user_embedding, company_embedding)
            scored_companies.append({"company": company, "score": score})

        # 5. 상위 3개 기업 선택
        top_companies = sorted(scored_companies, key=lambda x: x["score"], reverse=True)[:3]

        # 6. GPT에게 설명 요청
        top_descriptions = [f"{c['company']['name']} - {c['company']['description']}" for c in top_companies]
        final_prompt = f"""
다음은 사용자의 자기소개서와 유사한 상위 3개 기업입니다. 각 기업이 사용자에게 적합한 이유를 2~3문장으로 요약해서 자연스럽게 설명해줘.

자기소개서 키워드: {keyword_str}

기업 목록:
{chr(10).join(top_descriptions)}
"""
        explanation = get_gpt_reply(final_prompt)

        return render_template("index.html", response=explanation)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
