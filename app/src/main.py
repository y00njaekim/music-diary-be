import dotenv
import json
import traceback
import os
import requests
import datetime
import logging

from langchain.memory import ConversationSummaryMemory  # ConversationSummaryMemory(llm=llm, memory_key="history")
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

from llm_instance import llm
from analyzer.music import MusicAnalyzer
from chatbot.execute_state import execute_state, State, STATE_NEXT
from database.verification import verify_jwt
from database.manager import DBManager
from chatbot.lyrics_change import lyrics_change
import uuid  # 추가

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:3000", "https://music-diary-fe.vercel.app"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Authorization"],
            "supports_credentials": True,
        }
    },
    supports_credentials=True,
)


# 사용자별 메모리 저장소
user_memories = {}

# 사용자별 채팅봇 상태 저장소 (analyze_music에서 사용하던 것으로 추정)
chatbot_states = {}

# 임시 가사 저장소 (DB 연결 전 메모리 기반)
saved_lyrics: dict[str, dict] = {}

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")


# TODO 1: session (diary) 생성 및 DB 저장, session에 대한 초기 state, keyword row 생성 + session 정보 유지
# TODO 2: chat 진행에 따른 chat table update (insert), state, keyword table update (insert)
# TODO 3: lyrics, music, musicVis가 생성되거나 업데이트 되면 해당 table insert
# TODO 4: session이 끊길 때 summary table insert


@app.route("/analysis", methods=["POST"])
@verify_jwt
def analyze_music():
    # TODO: 임시 구현 - result.json 파일을 그대로 반환합니다.
    # try:
    #     with open("src/analyzer/a.json", "r", encoding="utf-8") as f:
    #         result = json.load(f)
    #     return jsonify(result), 200
    # except Exception as e:
    #     error_message = {"error": str(e), "traceback": traceback.format_exc()}
    #     print(json.dumps(error_message, indent=4))
    #     return jsonify(error_message), 500

    try:
        post_data = request.get_json()
        user_id = request.jwt_user["id"]
        if post_data is None:
            raise ValueError("No JSON data provided")

        print("Received data:", post_data)  # 요청 데이터 출력

        music_path = post_data.get("url")
        lyrics = post_data.get("lyrics")
        if not music_path:
            raise ValueError("Missing 'url' field")
        if not lyrics:
            raise ValueError("Missing 'lyrics' field")

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

        bpm = result["BPM"]
        instruments = result["Instruments"]  # 예: ["piano","drum"]
        emotions = result["Emotions"]  # 예: ["happy","excited"]

        # 리스트인 Instruments, Emotions를 문자열로 합치고, BPM을 포함해 하나의 문자열로 만듭니다.
        final_str = f"BPM: {bpm}, Instruments: {', '.join(instruments)}, Emotions: {', '.join(emotions)}"
        print(f"분석 요약: {final_str}")
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
        # 쿼리 파라미터에서 sid 추출 및 출력
        sid = request.args.get("sid")
        # print(f"[generate_response] sid: {sid}")
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
        if turn != 0 and not user_input:
            raise ValueError("input이 필요합니다")
        if not state:
            raise ValueError("state가 필요합니다")

        # print(f"요청 받음 - user_id: {user_id}, user_name: {user_name}, state: {state}, turn: {turn}")

        # 사용자별 메모리 가져오기 또는 생성
        if user_id not in user_memories:
            user_memories[user_id] = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)
            print(f"새로운 사용자 {user_id}에 대한 메모리 생성됨")

        memory = user_memories[user_id]

        if turn == 0:
            memory.clear()

        print(f"slot: {slot}")
        # execute_state 함수 실행
        try:
            state_enum = State(state)
        except ValueError:
            # 잘못된 state 값이 들어온 경우 에러 처리
            raise ValueError(f"알 수 없는 state 값입니다: {state}")

        response, flag, updated_slot = execute_state(user_input=user_input, state=state_enum, turn=turn, slot=slot, memory=memory)

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


@app.route("/lyrics/change", methods=["POST"])
@verify_jwt
def change_lyrics_api():
    """가사 변경 API"""
    try:
        post_data = request.get_json()
        if not post_data:
            return jsonify({"error": "JSON 데이터가 제공되지 않았습니다"}), 400

        total_lyrics = post_data.get("total_lyrics")
        change_lyrics = post_data.get("change_lyrics")
        user_lyric_prompt = post_data.get("user_lyric_prompt")

        if not all([total_lyrics, change_lyrics, user_lyric_prompt]):
            return jsonify({"error": "total_lyrics, change_lyrics, user_lyric_prompt 필드가 필요합니다"}), 400

        result_lyrics = lyrics_change(total_lyrics=total_lyrics, change_lyrics=change_lyrics, user_lyric_prompt=user_lyric_prompt)

        return jsonify({"changed_lyrics": result_lyrics}), 200

    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error in /lyrics/change: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


@app.route("/next_state", methods=["POST"])
@verify_jwt
def next_state():
    try:
        post_data = request.get_json()
        if post_data is None:
            raise ValueError("JSON 데이터가 제공되지 않았습니다")

        # JWT에서 추출한 사용자 정보
        jwt_user = request.jwt_user
        user_id = jwt_user["id"]

        # 현재 state만 추출
        state_str = post_data.get("state")

        if not state_str:
            raise ValueError("state가 필요합니다")

        # Enum 변환 및 다음 state 계산
        try:
            current_state = State(state_str)
        except ValueError:
            raise ValueError(f"알 수 없는 state: {state_str}")

        next_state_enum = STATE_NEXT.get(current_state)
        next_state_str = next_state_enum.value if next_state_enum else None

        response_data = {"state": next_state_str, "turn": 0}
        return jsonify(response_data), 200

    except ValueError as ve:
        error_message = {"error": str(ve)}
        print(f"ValueError: {str(ve)}")
        return jsonify(error_message), 400
    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


@app.route("/move_to_extraction_source", methods=["POST"])
@verify_jwt
def move_to_extraction_source():
    try:
        post_data = request.get_json()
        if post_data is None:
            raise ValueError("JSON 데이터가 제공되지 않았습니다")

        # JWT에서 추출한 사용자 정보
        jwt_user = request.jwt_user
        user_id = jwt_user["id"]

        # extraction_source 상태로 이동
        response_data = {"state": "extraction_source", "turn": 0}
        return jsonify(response_data), 200

    except ValueError as ve:
        error_message = {"error": str(ve)}
        print(f"ValueError: {str(ve)}")
        return jsonify(error_message), 400
    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


@app.route("/session/start", methods=["POST"])
@verify_jwt
def start_session():
    try:
        # JWT에서 추출한 사용자 정보
        jwt_user = request.jwt_user
        user_id = jwt_user["id"]

        # 세션 ID 생성
        session_id = str(uuid.uuid4())

        # 사용자별 세션 저장소에 저장 (선택적)
        if user_id not in user_memories:
            user_memories[user_id] = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)

        return jsonify({"sid": session_id, "user_id": user_id}), 200

    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error in /session/start: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


@app.route("/health", methods=["GET"])
def health_check():
    try:
        current_time = datetime.datetime.now().isoformat()
        return jsonify({"status": "healthy", "timestamp": current_time, "service": "music-diary-backend", "version": "1.0.0"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e), "timestamp": datetime.datetime.now().isoformat()}), 500


@app.route("/lyrics/save", methods=["POST"])
@verify_jwt
def save_lyrics_api():
    """사용자가 확정한 가사를 저장하는 API (임시: 메모리/파일 저장)"""
    try:
        # 세션 ID (선택) – 쿼리스트링으로 전달됨
        sid = request.args.get("sid") or "unknown_session"

        post_data = request.get_json()
        if not post_data:
            return jsonify({"error": "JSON 데이터가 제공되지 않았습니다"}), 400

        lyrics = post_data.get("lyrics")
        if not lyrics:
            return jsonify({"error": "lyrics 필드가 필요합니다"}), 400

        # JWT 사용자 정보
        jwt_user = request.jwt_user
        user_id = jwt_user["id"]

        # 메모리 저장 (DB 도입 전 임시)
        saved_lyrics.setdefault(user_id, {})[sid] = {
            "lyrics": lyrics,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        # 파일로도 백업 (optional)
        try:
            os.makedirs("saved_lyrics", exist_ok=True)
            file_path = os.path.join("saved_lyrics", f"{user_id}_{sid}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(lyrics)
        except Exception as fe:
            # 파일 저장 실패해도 치명적이지 않으므로 로그만 출력
            print(f"[save_lyrics_api] 파일 저장 실패: {fe}")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        error_message = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"Error in /lyrics/save: {json.dumps(error_message, indent=4)}")
        return jsonify(error_message), 500


if __name__ == "__main__":
    dotenv.load_dotenv()
    db_manager = DBManager()
    port = int(os.getenv("PORT", 5000))  # Render 환경 변수 PORT 사용
    app.run(host="0.0.0.0", port=port)
