import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Disable caching for real-time data updates & allow CORS
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
    url = f"http://localhost:{PORT}/user_pitching_analysis.html"
    
    print(f"=======================================================")
    print(f"  Science Pitching 로컬 웹 서버가 실행되었습니다.")
    print(f"  - 주소: http://localhost:{PORT}/")
    print(f"  - [분석 리포트] 피칭 분석 (Pitching Analysis): http://localhost:{PORT}/pitching_analysis.html")
    print(f"  - [시뮬레이터] 피칭 시각화 (Pitching Visualization): http://localhost:{PORT}/pitching_visualization.html")
    print(f"  - [사용자] 투구 분석 리포트: http://localhost:{PORT}/user_pitching_analysis.html")
    print(f"  (서버를 종료하려면 Ctrl+C를 누르세요)")
    print(f"=======================================================")
    
    # Auto-open browser if requested
    if "--open" in sys.argv or "-o" in sys.argv:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
