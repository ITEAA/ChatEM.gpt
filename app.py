from flask import Flask, jsonify
import requests
import os
import xml.etree.ElementTree as ET

app = Flask(__name__)

# Render 환경변수에서 불러오기
auth_key = os.getenv("JOB_API_KEY")

@app.route("/")
def test_kosme_api():
    url = "https://118.67.151.173/data/api/jopblancApi.do"
    params = {
        "authKey": auth_key,
        "callTp": "L",           # 목록 조회
        "listCount": 3           # 3개만 받아보기
    }

    try:
        response = requests.get(url, params=params, verify=False)  # SSL 인증서 무시 (IP 직접 호출 시 필요)
        xml_root = ET.fromstring(response.content)

        # XML 결과 중 일부만 추출해서 보여주기
        jobs = []
        for job in xml_root.findall(".//jobList"):
            jobs.append({
                "기업명": job.findtext("entrprsNm"),
                "공고명": job.findtext("pblancSj"),
                "근무지역": job.findtext("areaStr"),
                "고용형태": job.findtext("emplymStleSeStr")
            })

        return jsonify({"status": "success", "jobs": jobs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
