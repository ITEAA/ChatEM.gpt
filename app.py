from flask import Flask, jsonify
import requests
import urllib3
import os

app = Flask(__name__)

@app.route("/")
def get_kosme_data():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url = "https://118.67.151.173/data/api/jopblancApi.do"
    params = {
        "authKey": "fYL5gLDcPZ/iE6TB7Rmg1AnxisbHUUFMUuK8Am/MxcIC5+G2awO4kGH6CjFbgwAorXjRlhuqogcHGSEyLzQXdoOW2XonGbNFkASwL8QBm6FkiXgC/hHz+Jr/HAInzOPG",
        "callTp": "L",
        "listCount": 5,
        "query": "개발"
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=15)
        return jsonify({
            "요청 URL": response.url,
            "상태 코드": response.status_code,
            "응답 내용 (앞 1000자)": response.text[:1000]
        })

    except Exception as e:
        return jsonify({"오류": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
