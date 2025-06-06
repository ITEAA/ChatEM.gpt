from flask import Flask, request, jsonify
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

@app.route("/proxy")
def proxy():
    try:
        url = "https://118.67.151.173/data/api/jopblancApi.do"
        params = request.args.to_dict()
        response = requests.get(url, params=params, verify=False, timeout=10)
        return response.text, response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
