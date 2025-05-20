import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://118.67.151.173/data/api/jopblancApi.do"
params = {
    "authKey": "여기에_인증키_붙여넣기",
    "callTp": "L",
    "listCount": 5,
    "query": "개발"
}

response = requests.get(url, params=params, verify=False, timeout=15)

print("📡 요청 URL:", response.url)
print("🔍 상태 코드:", response.status_code)
print("📦 응답 내용 (앞 1000자):\n", response.text[:1000])
