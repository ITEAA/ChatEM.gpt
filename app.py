import requests
import xml.etree.ElementTree as ET
import urllib3
import os

# 인증서 오류 무시용 (KOSME API는 https인데 인증서 에러 있음)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# .env에서 불러오기 (직접 키를 쓸 경우 아래 주석 해제하고 작성)
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("JOB_API_KEY")  # 또는 직접 입력: "여기에_인증키_입력"
BASE_URL = "https://118.67.151.173/data/api/jopblancApi.do"

# 테스트용 요청 파라미터
params = {
    "authKey": API_KEY,
    "callTp": "L",        # 목록 호출
    "listCount": 5,
    "query": "개발"       # 테스트 키워드
}

try:
    response = requests.get(BASE_URL, params=params, verify=False, timeout=10)
    print("📡 요청 URL:", response.url)
    print("🔍 응답 코드:", response.status_code)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        jobs = root.findall(".//jobList")

        print(f"✅ 총 {len(jobs)}개 채용공고 수신")
        for job in jobs:
            print("📌", job.findtext("entrprsNm"), "|", job.findtext("pblancSj"), "|", job.findtext("areaStr"))
    else:
        print("❌ API 오류:", response.text)

except Exception as e:
    print("🚨 요청 실패:", e)
