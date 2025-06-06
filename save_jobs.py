import requests

url = "http://127.0.0.1:5001/proxy"
params = {
    "authKey": "fYL5gLDcPZ/iE6TB7Rmg1AnxisbHUUFMUuK8Am/MxcIC5+G2awO4kGH6CjFbgwAorXjRlhuqogcHGSEyLzQXdoOW2XonGbNFkASwL8QBm6FkiXgC/hHz+Jr/HAInzOPG",
    "callTp": "L",
    "listCount": 10,
    "query": "개발"
}

response = requests.get(url, params=params, verify=False)
print(response.text)
