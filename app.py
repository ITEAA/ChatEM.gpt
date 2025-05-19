from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)
job_api_key = os.getenv("JOB_API_KEY")

@app.route("/")  # ✅ 루트 URL에서 실행되도록 수정
def test_api():
    url = "https://118.67.151.173/data/api/jopblancApi.do"
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
