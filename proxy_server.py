from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/proxy')
def proxy():
    try:
        url = "https://118.67.151.173/data/api/jopblancApi.do"
        resp = requests.get(url, params=request.args, verify=False, timeout=10)
        return resp.text
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(port=5001)
