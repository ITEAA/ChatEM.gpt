import os
import openai
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from embedding import extract_keywords_with_gpt, match_resume_with_companies
from convert_xml_to_json import convert_company_xml_to_json
from save_jobs import fetch_company_list

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.form.get("message", "")
        file = request.files.get("file")

        # PDF 또는 텍스트 입력 처리
        if file:
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                file.save(tmp.name)
                text = extract_text_from_pdf(tmp.name)
                os.unlink(tmp.name)
        else:
            text = message

        if not text.strip():
            return jsonify({"reply": "❌ 이력서나 자기소개서를 입력하거나 파일을 업로드해주세요."})

        # 키워드 추출
        keywords = extract_keywords_with_gpt(text)
        print("🔍 추출된 키워드:", keywords)

        # 기업 리스트 불러오기 (더미 API 활용)
        companies_xml = fetch_company_list()
        companies = convert_company_xml_to_json(companies_xml)

        # 매칭 알고리즘 실행
        matched = match_resume_with_companies(text, companies)

        # GPT 프롬프트 생성
        prompt = generate_gpt_prompt(keywords, matched)

        # GPT에게 결과 생성 요청
        reply = get_gpt_reply(prompt)
        return jsonify({"reply": reply})

    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify({"reply": "❌ 서버 오류가 발생했습니다. 다시 시도해주세요."})

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print("❌ PDF 추출 오류:", e)
        return ""

def generate_gpt_prompt(keywords, matched_companies):
    prompt = """
당신은 사용자 이력서를 기반으로 맞춤 기업을 추천하는 AI입니다. 아래는 사용자 이력서에서 추출한 키워드와 추천된 기업들입니다.

[사용자 키워드]
"""
    prompt += ", ".join(keywords)
    prompt += """

[추천 기업 리스트]
"""
    if not matched_companies:
        prompt += "- 조건에 맞는 기업이 없습니다."
    else:
        for c in matched_companies:
            prompt += f"- 기업명: {c['corpNm']} / 주소: {c['address']} / 산업: {c['indTp']}\n"

    prompt += """

위의 내용을 바탕으로 사용자의 관심에 맞는 2~3개 회사를 선택해 친절하게 추천해 주세요. 각 회사가 왜 적합한지도 설명해 주세요.
"""
    return prompt

def get_gpt_reply(prompt):
    try:
        print("🧪 GPT 프롬프트 길이:", len(prompt))
        print("🔥 최종 GPT 프롬프트:\n", prompt[:1000], "... 생략")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 취업 지원자를 위해 회사를 추천해주는 전문가야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()
        return reply if reply else "❌ GPT 응답이 비어 있습니다."

    except Exception as e:
        print("❌ GPT 호출 오류:", e)
        return "❌ GPT 호출 중 오류가 발생했습니다. 다시 시도해주세요."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
