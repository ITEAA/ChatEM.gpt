from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)

# 환경변수에서 인증키 가져오기 (.env에서 읽히는 전제)
job_api_key = os.getenv("JOB_API_KEY")

@app.route("/test-api")
def test_api():
    url = "https://job.kosmes.or.kr/openApi/interestedJob/openApiJopblancList.do"
    params = {
        "serviceKey": job_api_key,
        "searchKeyword": "AI",
        "numOfRows": 3,
        "pageNo": 1,
        "returnType": "json"
    }
    response = requests.get(url, params=params)
    return jsonify({
        "status": response.status_code,
        "data": response.json() if response.status_code == 200 else response.text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
