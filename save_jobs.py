import requests
import xml.etree.ElementTree as ET

url = "http://127.0.0.1:5001/proxy"
params = {
    "authKey": "여기에_API_KEY_입력",
    "callTp": "L",
    "listCount": 10,
    "query": "개발"
}

try:
    response = requests.get(url, params=params, verify=False)
    root = ET.fromstring(response.content)
    print("=== 채용공고 목록 ===")
    for item in root.findall(".//jobList"):
        name = item.findtext("entrprsNm", "기업명 없음")
        title = item.findtext("pblancSj", "제목 없음")
        print(f"- {name}: {title}")
except Exception as e:
    print("❌ 오류 발생:", str(e))
