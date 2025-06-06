import requests

url = "https://118.67.151.173/data/api/jopblancApi.do"
params = {
    "authKey": "fYL5gLDcPZ/iE6TB7Rmg1AnxisbHUUFMUuK8Am/MxcIC5+G2awO4kGH6CjFbgwAorXjRlhuqogcHGSEyLzQXdoOW2XonGbNFkASwL8QBm6FkiXgC/hHz+Jr/HAInzOPG",
    "callTp": "L",
    "listCount": 10,
    "query": "개발"
}

response = requests.get(url, params=params, verify=False)

with open("job_data.xml", "w", encoding="utf-8") as f:
    f.write(response.text)
print("✅ API 응답 저장 완료: job_data.xml")
