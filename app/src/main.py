import dotenv
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import json
import traceback
from analyzer.music import MusicAnalyzer
import os
from chatbot.state_step_chat import ChatbotState, STATE_STEPS_ORDER,STEP_VAR_DESCRIPTIONS, generate_question_for_step, extract_reply_for_step, extract_name_with_llm, call_suno, save_chat_history, call_suno_lyrics
import requests
from io import BytesIO
import re
import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)


llm = ChatOpenAI(
model_name="gpt-4-turbo",  # 최신 모델 사용
temperature=0.7,
openai_api_key=api_key  # 환경변수 설정 가능
)

chatbot_states = {}


@app.route('/analysis', methods=['POST'])
def analyze_music():
    try:
        post_data = request.get_json()
        if post_data is None:
            raise ValueError("No JSON data provided")
        
        print("Received data:", post_data)  # 요청 데이터 출력

        music_path = post_data.get('url')
        if not music_path:
            raise ValueError("Missing 'url' field")
        
        user_id=post_data["currentUser"]
        context = chatbot_states[user_id]['context']
        lyrics = context['lyrics']

        print(f"Music path: {music_path}, Lyrics: {lyrics}")
        
         # Google Drive 링크 처리
        if "drive.google.com" in music_path:
            if "/file/d/" in music_path:
                file_id = music_path.split('/d/')[1].split('/')[0]
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
        
        user_id=post_data["currentUser"]
        context=chatbot_states[user_id]['context']
        bpm = result['BPM']
        instruments = result['Instruments']  # 예: ["piano","drum"]
        emotions = result['Emotions']        # 예: ["happy","excited"]

        # 리스트인 Instruments, Emotions를 문자열로 합치고, BPM을 포함해 하나의 문자열로 만듭니다.
        final_str = f"BPM: {bpm}, Instruments: {', '.join(instruments)}, Emotions: {', '.join(emotions)}"
        context["music_analysis"]=final_str
        # 임시 파일 삭제
        os.remove("temp_music_file.wav")

        return jsonify(result), 200
    except ValueError as ve:
        error_message = {
            'error': str(ve),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_message, indent=4))  # 로그에 상세 오류 정보 출력
        return jsonify(error_message), 400
    except Exception as e:
        error_message = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_message, indent=4))  # 로그에 상세 오류 정보 출력
        return jsonify(error_message), 400

@app.route('/set_user_name', methods=['POST'])
def set_user_name():
    """
    사용자가 처음 접근하면 챗봇이 "당신을 어떻게 부르면 될까요?"라고 질문함.
    이후 유저 입력을 받아 이름을 추출하여 반환.
    """
    data = request.get_json()
    # print(data)
    # print(call_suno_lyrics("hh"))
    # music_url=call_suno("가나다","가나다라마바사","천천히")
    user_input=data["userName"]
    user_id=data["currentUser"]
    # ✅ (1) 유저 입력이 없으면 챗봇이 먼저 질문
    if not user_input:
        return jsonify([{"role": "bot", "content": "그룹 이름을 알려줄래요?"}])

    # ✅ (2) 유저가 입력하면 LLM으로 이름 추출
    user_name = extract_name_with_llm(llm, user_input)

    # 이름이 인식되지 않으면 다시 요청
    if user_name == "Unknown":
        return jsonify({"error": "이름을 인식하지 못했습니다. 다시 입력해주세요."}), 400
    
    chatbot_states[user_id] = {
    "user_name": user_name,
    "current_state": ChatbotState.THERAPEUTIC_CONNECTION.value,
    "current_step": 0,
    "context": {"user_name": user_name,}
    }
    print(f"저장된 유저 이름: {chatbot_states[user_id]['user_name']}")

    return jsonify({"userName": f"좋아요, {user_name}! 그룹명이 참신한데요? 앞으로 {user_name}님이라고 부를게요."})


# ✅ (1) 질문 생성 (POST /chat/question)
@app.route('/chat/question', methods=['POST'])
def generate_question():
    """현재 단계의 질문을 생성하고 프론트로 보냄."""
    data = request.get_json()
    print(data) 
    user_id=data["currentUser"]

    print(user_id)
    chat_state = chatbot_states[user_id]
    current_state = chat_state["current_state"]
    current_step_index = chat_state["current_step"]
    context = chat_state["context"]
    
    # 🔹 현재 진행 중인 단계 가져오기
    steps = STATE_STEPS_ORDER[current_state]
    step_name = steps[current_step_index]
    print(current_state, step_name)

    context.setdefault("step_chat_history", {}).setdefault(step_name, "")
    
    # 🔹 질문 생성
    question_text = generate_question_for_step(llm, current_state, step_name, context)
    
    if current_state == ChatbotState.MUSIC_CREATION.value and step_name == "lyrics_gen":
        suno_gen_lyrics=call_suno_lyrics(question_text.content)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context["step_chat_history"][step_name] += f"\n[{timestamp}] bot: {suno_gen_lyrics}"
        context["chat_history"] = context.get("chat_history", "") + f"\n[{timestamp}] bot: {suno_gen_lyrics}"
        print("==========sunolyrics=========")
        print(suno_gen_lyrics)
        context["lyrics"]=suno_gen_lyrics
        return jsonify([{"role": "bot", "content": suno_gen_lyrics + '\n 가사를 이렇게 만들어보면 어떨까요?',"lyrics":1, "suno":suno_gen_lyrics}])

    
    if current_state == ChatbotState.MUSIC_CREATION.value and step_name == "lyrics_discussion" and context.get("lyrics_flag",0):
        feedback=question_text.content
        suno_gen_lyrics=call_suno_lyrics(feedback)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context["step_chat_history"][step_name] += f"\n[{timestamp}] bot: {suno_gen_lyrics}"
        context["chat_history"] = context.get("chat_history", "") + f"\n[{timestamp}] bot: {suno_gen_lyrics}"
        print("==========change=========")
        print(suno_gen_lyrics)
        context['lyrics_flag']="Unknown"
        context["lyrics"]=suno_gen_lyrics

        return jsonify([{"role": "bot", "content": suno_gen_lyrics + '\n 가사를 이렇게 만들어보면 어떨까요?', "lyrics":1, "suno":suno_gen_lyrics}])
    
    if current_state == ChatbotState.MUSIC_CREATION.value and step_name == "style_gen":
        return jsonify([{"role": "bot", "content": question_text.content + '\n 이런 컨셉은 어떤가요?'}])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context["step_chat_history"][step_name] += f"\n[{timestamp}] bot: {question_text.content}"
    context["chat_history"] = context.get("chat_history", "") + f"\n[{timestamp}] bot: {question_text.content}"

    # print(context)
    return jsonify([{"role": "bot", "content": question_text.content}])

# ✅ (2) 사용자 응답 처리 (POST /chat/response)
@app.route('/chat/response', methods=['POST'])
def process_response():
    """사용자 응답을 받아 변수 추출 & 다음 단계 진행"""
    data = request.get_json()
    print(data)
    user_id=data["currentUser"]

    user_input = data.get("message", "").strip()

    chat_state = chatbot_states[user_id]
    
    user_name = chat_state["user_name"]
    current_state = chat_state["current_state"]
    current_step_index = chat_state["current_step"]
    context = chat_state["context"]
    
    # 🔹 현재 진행 중인 단계 가져오기
    steps = STATE_STEPS_ORDER[current_state]
    step_name = steps[current_step_index]
    required_vars = list(STEP_VAR_DESCRIPTIONS[current_state][step_name].keys())
    print(steps, step_name)


    # 🔹 사용자 입력 저장
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context.setdefault("step_chat_history", {}).setdefault(step_name, "")
    context["step_chat_history"][step_name] += f"\n[{timestamp}] {user_name}: {user_input}"
    context["chat_history"] = context.get("chat_history", "") + f"\n[{timestamp}] {user_name}: {user_input}"


    if current_state == ChatbotState.MUSIC_CREATION.value and step_name == "lyrics_discussion":
            print("가사 재생성")
            example_prompt = PromptTemplate.from_template("""
                사용자의 답변: "{user_input}"
                출력: {output}
                """)

            examples = [
            {
                "user_input": "가사를 수정하고 싶어",
                "output": "1"
            },
            {
                "user_input": "좋아요",
                "output": "0"
            },
            {
                "user_input": "바꾸고 싶지 않아",
                "output": "0"
            },
            {
                "user_input": "수정 안해도 될 것 같아",
                "output": "0"
            },
            {
                "user_input": "가사의 ~~부분을 ~~라고 바꾸는게 좋을 것 같아",
                "output": "1"
            },
            {
                "user_input": "이렇게 하는게 더 나을 것 같아",
                "output": "1"},
                {
                "user_input": "~~을 ~~라고 바꿔줘",
                "output": "1"}
        ]
            prompt = FewShotPromptTemplate(
            examples=examples,              # 사용할 예제들
            example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
            prefix=""" 
                - 만약 사용자가 **생성된 가사에 대해 바꾸고 싶다면**, "1"을 단독 출력하세요.
                - 예시: "바꾸고 싶어", "마음에 안들어", "이 부분은 수정하고 싶어"

                - 만약 사용자가 **음악 수정이 필요 없다고 판단하면**, "0"을 단독 출력하세요.
                - 예시: "좋아요", "수정 안 해도 될 것 같아"

                - 출력은 반드시 **"0" 또는 "1"만 단독으로 출력**해야 합니다.""",
            suffix="사용자의 답변: {user_input}",          # 예제 뒤에 추가될 접미사
            input_variables=["user_input"],      # 입력 변수 지정
        )
            chain = prompt | llm
            output = chain.invoke({"user_input": user_input})
            match = re.search(r'\b[01]\b', output.content)

            if match:
                lyrics_flag = int(match.group())
                if lyrics_flag == 1:
                    context['lyrics_flag']=1
                    context['lyrics']="Unknown"

                    print("🔄 사용자 요청: 가사 수정")
                    return jsonify([{"role": "bot", "content": "음악 재생성 시작."}])
                            
    # 🔹 사용자 입력을 바탕으로 변수 추출
    extract_reply_for_step(llm, current_state, step_name, context, context["step_chat_history"][step_name])

    if current_state == ChatbotState.MUSIC_DISCUSSION.value and step_name == "music_recreation":
        print("🎯 Checking if the user wants to modify the music...")
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""
            사용자의 답변 "{user_input}"을 분석하세요.

            - 만약 사용자가 **생성된 음악에 대해 바꾸고 싶다면**, "1"을 단독 출력하세요.
            - 예시: "바꾸고 싶어", "마음에 안들어", "이 부분은 수정하고 싶어", "더 ~~할 것 같아요", "더 나을 것 같아요", "다시 만들어줘", "고쳐줘", "수정해줘"

            - 만약 사용자가 **음악 수정이 필요 없다고 판단하면**, "0"을 단독 출력하세요.
            - 예시: "좋아요", "수정 안 해도 될 것 같아"

            - 출력은 반드시 **"0" 또는 "1"만 단독으로 출력**해야 합니다.
            """
        )

        chain = prompt | llm
        output = chain.invoke({"user_input": user_input})
        match = re.search(r'\b[01]\b', output.content)

        if match:
            recreation_flag = int(match.group())
            if recreation_flag == 1:
                prompt = PromptTemplate(
                input_variables=["user_input"],
                template="""
                사용자의 답변 "{user_input}"을 분석하세요.

                - 만약 사용자가 **생성된 가사에 대해 바꾸고 싶다면**, "1"을 단독 출력하세요.
                - 예시: "가사를 바꾸고싶어", "가사의 ~~부분을 수정하고 싶어", 

                - 만약 사용자가 **생성된 음악에 대해 바꾸고 싶다면**, "0"을 단독 출력하세요.
                - 예시: "더 빠르게", "장르를 바꾸고싶어", "노래가 마음에 안들어", "악기를 바꾸고싶어", "느리게", "~~한 음색으로"

                - 출력은 반드시 **"0" 또는 "1"만 단독으로 출력**해야 합니다.
                """
                )

                chain = prompt | llm
                output = chain.invoke({"user_input": user_input})
                s_flag = re.search(r'\b[01]\b', output.content)
                state_flag = int(s_flag.group())

                if state_flag==1:
                    # 🔹 사용자가 "음악 수정 원함" → Music_Creation.making_lyrics 단계로 이동
                    chat_state["current_state"] = ChatbotState.MUSIC_CREATION.value
                    chat_state["current_step"] = STATE_STEPS_ORDER[ChatbotState.MUSIC_CREATION.value].index("making_lyrics")
                    print("🔄 사용자 요청: 이전 단계로 돌아감 → Music_Creation.making_lyrics")
                    ##근데 이렇게하면 변수를 다 초기화해야하나?
                    return jsonify([{"role": "bot", "content": "음악을 다시 조정해볼게요. 어떤 방향으로 수정할까요?"}])
                else: 
                    # 🔹 사용자가 "음악 수정 원함" → Music_Creation.making_concept 단계로 이동
                    chat_state["current_state"] = ChatbotState.MUSIC_CREATION.value
                    chat_state["current_step"] = STATE_STEPS_ORDER[ChatbotState.MUSIC_CREATION.value].index("making_music")
                    print("🔄 사용자 요청: 이전 단계로 돌아감 → Music_Creation.making_music")
                    ##근데 이렇게하면 변수를 다 초기화해야하나?
                    return jsonify([{"role": "bot", "content": "음악을 다시 조정해볼게요. 어떤 방향으로 수정할까요?"}])
    

    if current_state == ChatbotState.MUSIC_CREATION.value and step_name == "style_gen":
            music_title = context.get("title", "")
            music_lyrics = context.get("lyrics", "")
            music_components = context.get("music_component", "")

            if all([music_title, music_lyrics, music_components]):
                print("음악 생성 시작")
                prompt = PromptTemplate(
                input_variables=["music_components"],
                template="""
                {music_components} 의 내용과 더불어
                노래 구성요소(장르, 스타일, 빠르기, 악기, 분위기등등)을 만들어주세요.
            
                예시와 같이 키워드만 쉼표로 구분해서 출력합니다. 
                예시) 피아노, 밝게, 리듬

                아래와 같이는 절대 하지마세요. (단순 단어나열이 아닌 노래주제: < 이런식의 사용)
                노래주제: 스트레스 해소, 가사: 종이들이 바람에 날려 내 마음 속 무게도 함께 흩날려…, 
                요구조건: 락, 빠른 템포, 드럼만 사용, 강렬하고 희망적인 분위기, 락, 강렬, 희망적, 빠른 템포, 드럼

                [중요]
                150자내로 생성해야합니다.
                되도록 키워드로 간결하게 설명해주세요. 
                """
                )

                chain = prompt | llm
                output = chain.invoke({"music_components": music_components})
                music_prompt = output.content
                print('tag', music_prompt)
                context["style_prompt"]=music_prompt
                music_url=call_suno(music_title, music_lyrics, music_prompt)
                steps = STATE_STEPS_ORDER[current_state]
                # 현재 step_index
                cur_idx = chat_state["current_step"]

                # 만약 steps 내에서 다음 인덱스가 있으면 +1
                if cur_idx + 1 < len(steps):
                    chat_state["current_step"] += 1
                else:
                    # 다음 step이 없으면 다음 state로 넘어감
                    state_keys = list(STATE_STEPS_ORDER.keys())
                    current_state_index = state_keys.index(current_state)
                    if current_state_index + 1 < len(state_keys):
                        chat_state["current_state"] = state_keys[current_state_index + 1]
                        chat_state["current_step"] = 0
                
                return jsonify([{"role": "bot", "content":music_url }])
            else:
                print("음악을 생성하기 위해 필요한 정보가 부족합니다.")
    # 🔹 필요한 변수가 모두 채워졌는지 확인
    all_vars_filled = all(var in context and context[var] != "Unknown" for var in required_vars)

    if all_vars_filled:
        if current_step_index + 1 < len(steps):
            chat_state["current_step"] += 1  # 같은 상태에서 다음 단계로 이동
        else:
            # 🔹 다음 상태로 이동
            state_keys = list(STATE_STEPS_ORDER.keys())
            current_state_index = state_keys.index(current_state)
            if current_state_index + 1 < len(state_keys):
                chat_state["current_state"] = state_keys[current_state_index + 1]
                chat_state["current_step"] = 0  # 새로운 상태의 첫 스텝
            else:
                save_chat_history(context,user_name)


    return jsonify([{"role": "bot", "content": "응답을 처리했습니다. 다음 질문을 요청해주세요."}])


@app.route('/save_history', methods=['POST'])
def save_chat():
    data = request.get_json()
    print(data)
    user_id=data["currentUser"]
    
    chat_state = chatbot_states[user_id] 
    user_name = chat_state["user_name"]
    chat_history = chat_state["context"]["chat_history"]

    print('---save---')
    return jsonify([{"user_name":user_name,"history":chat_history}])

        
if __name__ == '__main__':
    dotenv.load_dotenv()
    port = int(os.getenv("PORT", 5000))  # Render 환경 변수 PORT 사용
    app.run(host="0.0.0.0", port=port)