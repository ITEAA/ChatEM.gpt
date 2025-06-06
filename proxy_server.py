from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/proxy')
def proxy():
    try:
        # 클라이언트 요청 파라미터 가져오기
        auth_key = request.args.get('authKey')
        call_tp = request.args.get('callTp')
        list_count = request.args.get('listCount', 10)
        query = request.args.get('query', '개발')

        # 중진공 API URL 및 파라미터 설정
        url = "https://118.67.151.173/data/api/jopblancApi.do"
        params = {
            "authKey": auth_key,
            "callTp": call_tp,
            "listCount": list_count,
            "query": query
        }

        # API 요청 (검증 비활성화 + 타임아웃 30초)
        response = requests.get(url, params=params, verify=False, timeout=30)
        content_type = response.headers.get('Content-Type', '')

        # JSON이면 JSON으로 응답, 아니면 텍스트로 응답
        if "application/json" in content_type:
            return jsonify(response.json())
        else:
            return response.text, response.status_code

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
