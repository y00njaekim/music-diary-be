from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt
from pydantic import BaseModel, Field
from typing import Optional
import os
import requests
import time
from dotenv import load_dotenv
import json

from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

# Mureka API 엔드포인트 및 API 키 설정
mureka_api_endpoint = "https://api.mureka.ai"
mureka_api_key = os.getenv("MUREKA_API_KEY")


def print_memory_summary(memory):
    print("\n===== 💬 요약된 memory 내용 =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[현재 저장된 요약 없음]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""

    music_information: Optional[str] = Field(default=None, description="Detailed information about the user's usual music activities")

    genre: Optional[str] = Field(default=None, description="The genre of music the user wants to create")

    instrument: Optional[str] = Field(default=None, description="Instruments to be included in the music")

    mood: Optional[str] = Field(default=None, description="The mood or atmosphere of the music")

    vocal: Optional[str] = Field(default=None, description="Information about the desired vocalist or vocal style")

    tempo: Optional[str] = Field(default=None, description="The tempo or speed of the music")

    title: Optional[str] = Field(default=None, description="The title of the music that you specified")


def music_making(user_input, llm, memory, pre_slot):
    extraction_source_question = f"""
    가사는 수정하지 않습니다.
    [Making Music Task]
    - 만들고 싶은 음악의 세부사항을 정합니다.
    - 사용자가 어려워하는 경우에는 추천하여 진행합니다.
    Example:
    - 위의 가사를 가지고 어떤 장르/스타일의 음악으로 만들면 좋을까요?
    - 노래의 빠르기는 어떻게 할까요? 
    - 반주 악기는 어떤 것을 선호하나요?
    - 음악의 전반적인 분위기는 어떻게 할까요? 
    - 가수의 음색은 어땠으면 좋을까요?
    - 가수의 성별은 어땠으면 좋을까요?
    - 이 주제에 맞는 음악요소를 추천 받고 싶나요? 
    - 노래의 제목은 무엇으로 하고 싶으신가요?
    """
    full_few_shot_dialogue = """
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 앞에서 이야기한 감성적인 분위기를 살려서 노래를 만든다면, 어떤 장르가 어울릴까요? 어쿠스틱 발라드, 포크 스타일 등 추천해드릴까요?
    Human: 어쿠스틱 발라드 좋을 것 같아요.
    AI: 노래의 빠르기는 느린 템포가 괜찮을까요, 아니면 약간 리듬감 있게 중간 정도로 가볼까요?
    Human: 너무 느리진 않고 중간 템포였으면 해요.
    AI: 반주에 어떤 악기를 중심으로 하면 좋을까요? 피아노, 기타, 현악기 등이 있어요. 피아노: 감성을 섬세하게 표현할 수 있어요. 차분하고 잔잔한 분위기에 잘 어울려요. 기타: 따뜻하고 편안한 느낌이에요. 어쿠스틱이나 인디 감성과 잘 맞아요. 현악기: 깊고 풍부한 울림이 있어요. 감정을 더 감성적으로 전달할 수 있어요. 어떤 악기가 지금의 분위기와 가장 잘 어울릴까요?
    Human: 어쿠스틱 기타랑 피아노 조합이 좋을 것 같아요.
    AI: 음악의 분위기는 따뜻하고 위로가 되는 느낌일까요, 아니면 더 쓸쓸하고 내면적인 느낌일까요?
    Human: 따뜻하고 위로가 되는 분위기로 하고 싶어요.
    AI: 좋습니다! 마지막으로, 노래를 부를 가수의 음색과 성별은 어떻게 상상하고 계세요?
    Human: 감성적이고 맑은 음색이면 좋겠어요. 여성보컬이면 더 좋아요. 저는 여성보컬은 조금 더 잘 들리거든요.
    AI: 그럼 노래 제목은 무엇으로 하는 것이 좋을까요?
    Human: 위로의 말로 하는게 좋을 것 같아요.
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + extraction_source_question
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "아래를 보고 참고하여 질문을 생성하세요."
        + "Previous Slot: {pre_slot}\n"
        + "Chat history: {history}\n"
        + "User said: {user_message}",
    )

    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")



    question_chain = question_prompt | llm | StrOutputParser()
    question = question_chain.invoke({"user_message": user_input, "history": history,"pre_slot": pre_slot})

    # print_memory_summary(memory)
    memory.save_context({"input": user_input}, {"output": question})

    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return question, slot


def query_mureka_task(id: str):
    """지정된 ID의 작업 상태를 Mureka API에 조회합니다."""
    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
    }
    response = requests.get(mureka_api_endpoint + f"/v1/song/query/{id}", headers=headers)
    response.raise_for_status()
    return response.json()


def generate_mureka_song_and_wait(title: str, lyrics: str, music_component: str) -> str:
    """
    Mureka API에 노래 생성을 요청하고, 작업이 완료될 때까지 대기한 후
    오디오 URL을 반환합니다.
    """
    print(f"제목: {title}")
    print(f"음악 스타일: {music_component}")

    # 1. 노래 생성 요청 (POST)
    headers = {"Authorization": f"Bearer {mureka_api_key}", "Content-Type": "application/json"}
    payload = {
        "lyrics": lyrics,
        "model": "auto",
        "prompt": music_component,
    }

    try:
        response = requests.post(mureka_api_endpoint + "/v1/song/generate", headers=headers, json=payload, timeout=(5, 60))
        response.raise_for_status()

        res_data = response.json()
        task_id = res_data.get("id")
        print(f"노래 생성 작업 시작. 작업 ID: {task_id}")

        # 2. 작업 완료까지 대기 (while 루프)
        retry_delay = 5  # 5초마다 상태 확인
        max_retries = 100  # 최대 100번 시도 (약 8분)
        retry_count = 0

        while retry_count < max_retries:
            task_status_response = query_mureka_task(task_id)
            status = task_status_response.get("status")

            if status == "succeeded":
                audio_url = task_status_response["choices"][0]["url"]
                print(f"노래 생성 성공! 오디오 URL: {audio_url}")
                return audio_url
            elif status == "failed":
                print(f"작업 실패: {task_status_response}")
                return "Task failed"
            else:
                # 상태가 'processing' 이거나 다른 상태일 경우
                print(f"작업 진행 중... (상태: {status}). {retry_delay}초 후 다시 시도합니다.")
                time.sleep(retry_delay)
                retry_count += 1

        print("최대 시도 횟수를 초과했습니다. 작업 시간 초과.")
        return "Task timed out"

    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return f"API Error: {e}"


def music_creation(user_input, llm, memory, pre_slot):
    """
    CombinedSlot(dict) 타입의 user_input에서 가사와 음악 스타일 정보를 추출하여
    Mureka API로 음악을 생성하고, 오디오 URL을 반환합니다.
    """
    # user_input이 문자열이면 딕셔너리로 파싱
    if isinstance(user_input, str):
        user_input_dict = json.loads(user_input)
    # 1. 가사 추출
    lyrics = user_input_dict.get("lyrics", None)
    if not lyrics:
        response = "가사가 입력되지 않았습니다."
        # history 저장 및 slot 생성
        memory_vars = memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        structured_llm = llm.with_structured_output(schema=OutputFormat)
        slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
        slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))
        return response, slot

    # 2. 음악 스타일 프롬프트 생성
    style_elements = []
    for key in ["genre", "instrument", "mood", "vocal", "tempo"]:
        value = user_input_dict.get(key, None)
        if value:
            # TODO: mureka에 meta tag 넣을 때 key: value 구조가 아닐텐데? 그냥 tag1, tag2, ... 이렇게 넣을 거임.
            # => 넵 
            # style_elements.append(f"{key}: {value}")
            style_elements.append(value)
    music_component = ", ".join(style_elements) if style_elements else ""

    # 3. 제목 추출 (없으면 'Untitled Song')
    # TODO: slot에서 name은 user name 아닌가?
    # => title로 업데이트!
    title = user_input_dict.get("title", "Untitled Song")

    # 4. Mureka API 호출 및 결과 반환
    audio_url = generate_mureka_song_and_wait(title, lyrics, music_component)

    if audio_url.startswith("http"):
        response = f"노래가 성공적으로 생성되었습니다!\n오디오 파일: {audio_url}"
    else:
        response = f"노래 생성에 실패했습니다: {audio_url}"

    # history 저장
    memory.save_context({"input": user_input}, {"output": response})
    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")

    # structured LLM으로 slot 생성
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return response, slot
