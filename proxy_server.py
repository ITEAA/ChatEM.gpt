from flask import Flask, request, Response
import requests

app = Flask(__name__)

@app.route("/proxy", methods=["GET"])
def proxy():
    api_url = "https://118.67.151.173/data/api/jopblancApi.do"
    params = request.args

    try:
        response = requests.get(api_url, params=params, verify=False, timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get("Content-Type"))
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
