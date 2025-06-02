import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
from analyzer.music import MusicAnalyzer
import os
import requests
import datetime
from langchain.memory import ConversationSummaryMemory  # ConversationSummaryMemory(llm=llm, memory_key="history")
from langchain_openai import ChatOpenAI
from chatbot.execute_state import execute_state
import jwt
from functools import wraps
import logging

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:3000", "https://your-production-domain.com"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Authorization"],
            "supports_credentials": True,
        }
    },
    supports_credentials=True,
)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 사용자별 메모리 저장소
user_memories = {}

# 사용자별 채팅봇 상태 저장소 (analyze_music에서 사용하던 것으로 추정)
chatbot_states = {}

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")


def verify_jwt(f):
    """JWT 토큰을 검증하는 데코레이터 (대칭키 방식)"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "인증 헤더가 없습니다"}), 401

        if not SUPABASE_JWT_SECRET:
            return jsonify({"error": "서버 설정 오류: JWT 시크릿이 없습니다."}), 500

        try:
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "잘못된 인증 헤더 형식입니다"}), 401
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
            user_id = payload.get("sub")  # user.id
            email = payload.get("email", "")  # user.email
            user_metadata = payload.get("user_metadata", {})
            name = user_metadata.get("name", email.split("@")[0] if email else "User")
            app_metadata = payload.get("app_metadata", {})
            request.jwt_user = {"id": user_id, "email": email, "name": name, "user_metadata": user_metadata, "app_metadata": app_metadata}
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "토큰이 만료되었습니다"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": "유효하지 않은 토큰입니다"}), 401
        except Exception as e:
            return jsonify({"error": f"토큰 검증 중 오류: {str(e)}"}), 500

    return decorated_function


@app.route("/analysis", methods=["POST"])
@verify_jwt
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


@app.route("/generate_response", methods=["POST"])
@verify_jwt
def generate_response():
    try:
        # POST 데이터 가져오기
        post_data = request.get_json()
        if post_data is None:
            raise ValueError("JSON 데이터가 제공되지 않았습니다")

        # JWT에서 추출한 사용자 정보
        jwt_user = request.jwt_user
        user_id = jwt_user["id"]
        user_name = jwt_user["name"]
        user_email = jwt_user["email"]

        # 필수 파라미터 추출
        user_input = post_data.get("input")
        state = post_data.get("state")
        turn = post_data.get("turn", 0)
        slot = post_data.get("slot", {})

        # slot에 사용자 이름 추가
        if user_name:
            slot["name"] = str(user_name)

        # 필수 파라미터 검증
        if not user_input:
            raise ValueError("input이 필요합니다")
        if not state:
            raise ValueError("state가 필요합니다")

        print(f"요청 받음 - user_id: {user_id}, user_name: {user_name}, state: {state}, turn: {turn}")

        # 사용자별 메모리 가져오기 또는 생성
        if user_id not in user_memories:
            user_memories[user_id] = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)
            print(f"새로운 사용자 {user_id}에 대한 메모리 생성됨")

        memory = user_memories[user_id]

        print(f"slot: {slot}")
        # execute_state 함수 실행
        response, flag, updated_slot = execute_state(user_input=user_input, state=state, turn=turn, slot=slot, memory=memory)

        # 대화 컨텍스트를 메모리에 저장 (save_context 메서드 사용)
        # 이렇게 하면 자동으로 요약이 업데이트됩니다
        memory.save_context(inputs={"input": user_input}, outputs={"output": response})

        # turn 증가
        next_turn = turn + 1

        # 응답 데이터 구성
        response_data = {"response": response, "turn": next_turn, "state": state, "slot": updated_slot, "flag": flag}

        print(f"응답 생성 완료 - turn: {next_turn}, flag: {flag}")

        return jsonify(response_data), 200

    except ValueError as ve:
        error_message = {"error": str(ve)}
        print(f"ValueError: {str(ve)}")
        return jsonify(error_message), 400
    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


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
