import requests
import xml.etree.ElementTree as ET
import json

url = "http://127.0.0.1:5001/proxy"
params = {
    "authKey": "여기에_API_KEY_입력",
    "callTp": "L",
    "listCount": 5,
    "query": "AI"
}

try:
    response = requests.get(url, params=params, verify=False)
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall(".//jobList"):
        results.append({
            "company": item.findtext("entrprsNm", "기업명 없음"),
            "title": item.findtext("pblancSj", "제목 없음"),
            "location": item.findtext("areaStr", ""),
            "tags": [item.findtext("dtyStr", ""), item.findtext("emplymStleSeStr", "")]
        })
    print(json.dumps(results, indent=2, ensure_ascii=False))
except Exception as e:
    print("❌ 오류 발생:", str(e))
