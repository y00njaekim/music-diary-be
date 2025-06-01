import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
from analyzer.music import MusicAnalyzer
import os
import requests
import datetime

app = Flask(__name__)
CORS(app)


@app.route("/analysis", methods=["POST"])
def analyze_music():
    try:
        post_data = request.get_json()
        if post_data is None:
            raise ValueError("No JSON data provided")

        print("Received data:", post_data)  # 요청 데이터 출력

        music_path = post_data.get("url")
        if not music_path:
            raise ValueError("Missing 'url' field")

        user_id = post_data["currentUser"]
        context = chatbot_states[user_id]["context"]
        lyrics = context["lyrics"]

        print(f"Music path: {music_path}, Lyrics: {lyrics}")

        # Google Drive 링크 처리
        if "drive.google.com" in music_path:
            if "/file/d/" in music_path:
                file_id = music_path.split("/d/")[1].split("/")[0]
            elif "id=" in music_path:
                file_id = music_path.split("id=")[1].split("&")[0]
            else:
                raise ValueError("Invalid Google Drive URL")
            music_path = f"https://drive.google.com/uc?id={file_id}&export=download"
            print(f"Converted Google Drive link to direct download URL: {music_path}")

        # 파일 다운로드
        response = requests.get(music_path, stream=True)
        if response.status_code != 200:
            raise ValueError("Failed to download the music file")

        # 파일을 로컬에 임시 저장
        with open("temp_music_file.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        la = MusicAnalyzer("temp_music_file.wav", lyrics)
        la.analyze()
        result = la.get_final_format()

        user_id = post_data["currentUser"]
        context = chatbot_states[user_id]["context"]
        bpm = result["BPM"]
        instruments = result["Instruments"]  # 예: ["piano","drum"]
        emotions = result["Emotions"]  # 예: ["happy","excited"]

        # 리스트인 Instruments, Emotions를 문자열로 합치고, BPM을 포함해 하나의 문자열로 만듭니다.
        final_str = f"BPM: {bpm}, Instruments: {', '.join(instruments)}, Emotions: {', '.join(emotions)}"
        context["music_analysis"] = final_str
        # 임시 파일 삭제
        os.remove("temp_music_file.wav")

        return jsonify(result), 200
    except ValueError as ve:
        error_message = {"error": str(ve), "traceback": traceback.format_exc()}
        print(json.dumps(error_message, indent=4))  # 로그에 상세 오류 정보 출력
        return jsonify(error_message), 400
    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(error_message, indent=4))  # 로그에 상세 오류 정보 출력
        return jsonify(error_message), 400


@app.route("/health", methods=["GET"])
def health_check():
    try:
        current_time = datetime.datetime.now().isoformat()
        return jsonify({"status": "healthy", "timestamp": current_time, "service": "music-diary-backend", "version": "1.0.0"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e), "timestamp": datetime.datetime.now().isoformat()}), 500


if __name__ == "__main__":
    dotenv.load_dotenv()
    port = int(os.getenv("PORT", 5000))  # Render 환경 변수 PORT 사용
    app.run(host="0.0.0.0", port=port)
