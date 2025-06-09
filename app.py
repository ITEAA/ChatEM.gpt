# ✅ 최종 app.py (배포 가능)
# - TF-IDF로 후보 기업 추출
# - KoBERT로 의미 기반 유사도 계산
# - GPT로 각 기업 설명 생성

import os
import json
import fitz
import openai
import torch
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key"
GG_CACHED_FILE = "gg_employment_cached.json"
user_states = {}

# KoBERT 로딩
model = None
try:
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    model = AutoModel.from_pretrained("monologg/kobert", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    print("[❌ KoBERT 로드 실패]", e)

# 기업 정보 불러오기
try:
    with open(GG_CACHED_FILE, "r", encoding="utf-8") as f:
        cached_companies = json.load(f)
except:
    cached_companies = []

# PDF 텍스트 추출
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([p.get_text() for p in doc]).strip()

# GPT 키워드 추출
def extract_keywords(text):
    prompt = f"""
    다음 자기소개서 또는 이력서에서 핵심 키워드를 추출해줘.
    - 5~10개 정도 뽑아줘.
    - 키워드는 콤마(,)로 구분해서 출력해줘.

    내용:
    {text}
    """
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return [kw.strip() for kw in res.choices[0].message.content.split(",") if kw.strip()]
    except:
        return []

# TF-IDF 필터링
def tfidf_filter(user_text, companies, top_n=30):
    corpus = [user_text] + [c.get("채용공고명", "") + " " + c.get("회사명", "") for c in companies]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sim_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked = [(companies[i], sim_scores[i]) for i in range(len(companies))]
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    top_companies = [c for c, score in ranked[:top_n] if score > 0]
    return top_companies if len(top_companies) >= 10 else companies

# KoBERT 임베딩
def get_kobert_embedding(text):
    if model is None or not text:
        return torch.zeros(768).to(device)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    except:
        return torch.zeros(768).to(device)

# KoBERT 유사도 계산
def kobert_similarity(user_text, companies):
    ue = get_kobert_embedding(user_text).cpu().numpy().reshape(1, -1)
    results = []
    for c in companies:
        desc = c.get("채용공고명", "") + " " + c.get("회사명", "")
        ce = get_kobert_embedding(desc).cpu().numpy().reshape(1, -1)
        score = cosine_similarity(ue, ce).flatten()[0]
        results.append((c, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# GPT 설명 생성
def generate_reason_individual(user_text, company, score):
    prompt = f"""
    너는 지금부터 사용자의 특성과 선호도를 파악해 가장 적합한 기업을 추천해주는 역할을 수행한다. 다음 지침을 따라:

    1. 동작 모드
    - 사용자가 이력서를 제공했으므로 분석 모드로 자동 진입
    - 사용자의 자기소개서 내용과 해당 기업 정보를 기반으로 추천 이유를 분석하고 제공한다

    2. 금지된 표현 사용 금지
    - "분석모드입니다", "모드를 전환합니다" 등 시스템 작동 문구는 절대 사용하지 마라
    - 실패 시에는 "현재 정보만으로는 분석이 어렵습니다" 등 자연스러운 안내만 사용한다

    3. 대화 스타일
    - 친절하고 전문적인 어조로 작성하되, 설명은 자연스럽고 명확하게 전달하라

    아래 정보를 바탕으로, 사용자의 자기소개서 내용을 요약하고, 기업과 어떻게 연결되는지 설명해줘.

    [기업 정보]
    - 기업명: {company.get('회사명')}
    - 채용공고명: {company.get('채용공고명')}
    - 유사도 점수: {round(score, 2)}

    [사용자 자기소개서]
    {user_text}

    [설명 시작]
    """
    try:
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("GPT 설명 생성 실패:", e)
        return "설명을 생성하는 데 실패했습니다."

def recommend():
    data = request.json
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "자기소개서 내용이 없습니다."}), 400

    keywords = extract_keywords(user_text)
    candidates = tfidf_filter(user_text, cached_companies)
    scored = kobert_similarity(user_text, candidates)

    top_recommendations = []
    shown = set()
    for comp, sim in scored:
        if sim > 0 and (comp.get("회사명"), comp.get("채용공고명")) not in shown:
            shown.add((comp.get("회사명"), comp.get("채용공고명")))
            reason = generate_reason_individual(user_text, comp, sim)
            top_recommendations.append({
                "회사명": comp.get("회사명"),
                "채용공고명": comp.get("채용공고명"),
                "유사도": round(sim, 3),
                "추천이유": reason
            })
        if len(top_recommendations) >= 3:
            break

    return jsonify({"추천결과": top_recommendations})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
