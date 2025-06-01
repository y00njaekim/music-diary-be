import os
import requests
import time
import json
from typing import Dict, Any, List
from enum import Enum
import datetime
import requests
from requests.exceptions import RequestException, ChunkedEncodingError


# langchain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
# suno_end_point = os.getenv("SUNO_END_POINT")
suno_end_point = 'https://api.mureka.ai'
mureka_api_key = os.getenv("MUREKA_API_KEY")

################################################
# (A) State/Step 구조 & 변수 설명
################################################

class ChatbotState(Enum):
    THERAPEUTIC_CONNECTION = "Therapeutic_Connection"
    MUSIC_CREATION = "Music_Creation"
    MUSIC_DISCUSSION = "Music_Discussion"
    WRAP_UP = "Wrap_Up"

# [예시] 각 스텝에서 필요한 변수를 "설명"과 함께 정의
# 실제로는 한국어 설명을 붙이거나, 변수명을 더 다양하게 구성하시면 됩니다.
STEP_VAR_DESCRIPTIONS = {
    # 1) Therapeutic_Connection
    ChatbotState.THERAPEUTIC_CONNECTION.value: {
        "rapport_building": {
            "user_ready": "사용자의 음악만들기에 대한 관심 여부"
        },
        # "goal_and_motivation_building": {
        #     "motivation": "사용자가 음악만들기 활동을 통해 달성하고 싶은 목표",
        #     "difficulty": "사용자가 현재 겪고 있는 어려움, 어려움으로 야기되는 문제점",
        #     "emotion": "사용자가 최근 들어 많이 느끼는 감정"
        # },
        # "music_preference": {
        #     "music_info": "사용자가 좋아하거나 관심있거나 싫어하는 음악 정보 (장르, 스타일 등)"
        # },
    },

    # 2) Music_Creation
    ChatbotState.MUSIC_CREATION.value: {
        "making_concept": {
            "concept": "음악의 전반적인 컨셉 (분위기, 테마, 메시지 등)"
        },
        "making_lyrics": {
            "lyrics_keyword": "사용자가 작성한 대답 중 2개 이상의 가사에 들어갈 핵심 키워드 또는 아이디어",
            "lyrics_sentence":"사용자가 작성한 가사의 핵심 문장(3문장 이상)",
            "lyrics_flow": "사용자가 정한 가사의 흐름",
        },
        "lyrics_gen":{
            "lyrics": "생성된 최종 가사"
        },
        "lyrics_discussion": {
            "discussion_feedback": "가사에 대한 사용자 의견",
            "lyrics_flag":"가사변경 유무"

        },
        "making_music": {
            "title": "사용자가 만들 노래 제목",
            "music_component": "멜로디, 코드 진행, 리듬 등 구체적인 음악 아이디어",
            
        },
        "style_gen":{
            "style_prompt":"챗봇이 만들어준 노래 구성요소 프롬프트(어떤가요? < 이거 제외 / 단어로 구성)"
        }
    },

    # 3) Music_Discussion
    ChatbotState.MUSIC_DISCUSSION.value: {
        "music_opinion": {
            "individual_emotion": "사용자가 음악을 듣고 느낀 개인적 감정",
            "strength": "음악만들기 활동을 통해 느낀 사용자의 장점"
        },
        "music_recreation": {
            "change_music": "사용자가 바꾸고 싶거나 개선하고 싶은 음악 요소"
        },
    },

    # 4) Wrap_Up
    ChatbotState.WRAP_UP.value: {
        "reflection": {
            "change_mind": "음악 만들기 과정을 통해 얻은 강점"
        },
        "complete": {
            "feeling": "음악 만들기 활동을 통해서 사용자가 느낀 감정"
        },
    },
}

# (스텝별 메인 대화 프롬프트: 실제로는 더 풍부하게 작성 가능)
STEP_MAIN_PROMPTS = {
    ChatbotState.THERAPEUTIC_CONNECTION.value: {
        "rapport_building": """
            [라포 형성] 
            사용자와 음악에 대한 간단한 대화를 나누세요.
            예시)
            - 오늘 음악만들기 활동이 기대되시나요?
            - 음악만들기를 통해서 즐거운 시간을 보낼 준비 되셨나요?
            """,
        # "goal_and_motivation_building": """
        #     [목표/동기 파악] 
        #     사용자의 동기(motivation), 어려움(difficulty), 감정(emotion)을 파악하세요.
        #     현재 단계(Goal and Motivation-building)에서는 다음 정보가 필요합니다:

        #     1) difficulty (어려움)
        #     - 사용자가 생활 속에서 느끼는 가장 큰 어려움, 구체적인 상황(주제) + 그로 인한 문제점
        #         예: "직장 내 갈등으로 스트레스를 많이 받음" + "잠을 잘 못 자고 의욕이 떨어짐"
        #     - 만약 사용자가 '말하기 어렵다'거나 '잘 모르겠다'라고 하면, 
        #         대신 어떤 감정을 요즘 주로 느끼는지(emotion)를 파악하세요.

        #     2) emotion (감정) -> difficulty를 구체적으로 채웠을 경우 생략가능.
        #     - difficulty를 상세히 말하기 어려워하는 경우, 
        #         "최근 들어 주로 느끼는 감정이 있다면 어떤 것인가요?" 라고 추가 질문하세요.
        #     - 둘 다 말할 수 있으면 둘 다 수집해도 좋습니다.

        #     3) motivation (음악치료를 통해 얻고 싶은 것)
        #     - "기존에 듣던 음악에서 위로받은 경험이 있나요?" 라고 물어보고,
        #         - 만약 그렇다면 "어떤 경험이었는지" 묻고, 그 내용을 motivation에 반영하세요.
        #         - 만약 없다면 "음악치료로 무엇을 기대하는지", "어떤 목표가 있는지" 묻고 그 내용을 motivation에 담으세요.
        #     - 예: "음악을 통해 감정을 표현하고 싶다", "내면에 직면하고 싶다", "고립감을 해소하고 싶다" 등등

        #     [중요] 
        #     - 사용자가 답을 잘 못하면, 예시나 선택지를 제시할 수도 있습니다. 
        #     (예: “외부 문제(직장/인간관계), 내부 문제(성격/외모) 등이 있을까요?”)
        #     - 모든 질문을 한 번에 다 하지 말고, 사용자의 응답을 들은 뒤 추가 질문을 자연스럽게 이어가세요.
        #     - 사용자의 대답을 들으면 무조건 공감을 하고 대답을 해주세요.
        #     - 예시는 사용자가 대답을 망설일때만 제시하세요. 처음부터 제시하지 마세요. 
        #     """,
        # "music_preference": """
        #     [음악 선호] 
        #     사용자의 음악 선호(music_info)를 파악하세요.
        #     아래의 대화흐름으로 사용자에게 질문하고 대화를 진행해주세요: 

        #     1. goal_and_motivation_building단계에서 진행한 대화를 바탕으로 도입 질문을 시작하세요.
        #         - 이 difficulty를 해결하기 위해 음악을 사용한적이 있나요?
        #         - 노래를 들으며 difficulty를 치유한적이 있나요?
        #     2-1. 만약 '그렇다'고 응답하면 아래의 예시와 같은 질문을 진행하세요. (1~2개)
        #         - 그렇다면 그럴때 주로 감상하는 음악은 무엇인가요? 
        #         - 어떤 음악 활동(음악감상, 악기연주, 노래부르기 등)을 했나요?
        #         - 
        #     2-2. 만약 '그렇지 않다, 그런적없다'고 응답한다면, 아래와 같은 질문을 진행하세요. (1개)
        #         - 음악을 통해서 해결해보는 건 어떨까요? 특별히 원하는 음악이 있나요?
        #         - 음악은 좋은 해결수단이 될 수 있어요. 혹시 싫어하는 음악이 있나요?
            
        #     [중요] 
        #     - 너무 깊게 들어가지 않아도 됩니다. 사용자의 말에 공감이 가장 중요하다는 것 잊지마세요. 
        #     - 모든 질문을 한 번에 다 하지 말고, 사용자의 응답을 들은 뒤 추가 질문을 자연스럽게 이어가세요.
        #     - 질문을 3턴이상 하지마세요. 마무리는 공감으로 진행하세요.
        #     - 변수를 다 채우고 음악에 대한 공감을 진행하세요.
        # """,
    },
    ChatbotState.MUSIC_CREATION.value: {
        "making_concept": """
            [컨셉 설정] 
            곡(음악)의 전반적인 컨셉(concept)을 구체화하세요.
            사용자가 어려워하면 이전의 대화 기록을 통해 추천해주세요.
            "없어요", "모르겠어요"와 같은 반응이 나오면 이전의 대화에서 추천해준 후 이 단계를 끝내세요. 
            똑같은 질문을 반복하지 마세요.
            

            아래와 같은 대화 흐름을 따르세요. 한꺼번에 질문하지 말고 단계별로 하나씩 질문하세요. 
            (0) music info에 대해 공감
            - (좋아하는 음악에 대해 대화 나눈 경우) ~~한 음악를 좋아하시는 군요. ~~와 같은 음악을 만들어보는게 좋을 것 같아요.
            - ~~ 느낌의 음악를 싫어하실 수 있죠. 다른 스타일의 음악으로 만들어봐요.

            (1) 주제 설정하기
            - {difficulty}에서 이야기한 주제를 바탕으로 음악을 만들어볼까요?
            - 어떤 감정이나 상황을 음악으로 담고 싶나요?

            (2) 이야기 구체화하기
            - 이 노래 안에 어떤 이야기를 담고 싶나요?
            - 그 이야기를 담고 싶은 이유가 있나요?
            - 이 음악이 어떤 감정을 전달했으면 좋겠나요?
            
            사용자가 어려움을 겪거나 모르겠다고 대답할때만 아래와 같은 응답을 진행하세요. 
             - 예시를 제시하세요:
            "예를 들면, 극복하고 싶은 어려움, 행복했던 순간, 위로받고 싶은 감정 등이 있을까요?"
            - 선택지를 제안할 수도 있어요:
            "예를 들어, ‘외로움’, ‘성장’, ‘추억’, ‘희망’ 같은 주제도 가능해요."
            """,
        "making_lyrics": """
            [가사 작성] 
            사용자가 가사의 키워드, 문장, 흐름을 이야기하면서 사용자의 이야기가 담긴 가사를 만들도록 유도합니다.
            처음부터 예시를 들어주지 마세요. 어려워하는 경우에만 예시를 들어 사용자를 도와줍니다. 
            아래의 (1),(2),(3)의 과정을 거쳐서 사용자의 응답을 받아야합니다.
            사용자의 생각과 감정을 깊이 반영할 수 있도록, 다양한 질문과 선택지를 통해 가사 제작 과정을 이끌어갑니다.

            (1) 가사 키워드 도출하기 — 감정과 이미지 확장하기
            사용자에게 충분한 선택지를 제공하고 구체적인 이미지를 떠올릴 수 있도록 질문하세요.

            기본 질문:
            이전에 정한 {concept} 를 음악으로 표현한다면, 어떤 이미지가 떠오르시나요?
            이 주제를 떠올릴 때 가장 먼저 드는 감정이나 단어가 있나요?
            그 감정을 더 구체적으로 표현하면 어떤 단어가 떠오르시나요?
            머릿속에 그려지는 장면이나 특정 이미지가 있나요?

            사용자가 어려워할 때:
            - 예시 단어를 제시하세요:
            “예를 들어, ‘희망’, ‘눈물’, ‘바람’, ‘기다림’, ‘돌아봄’ 같은 단어들이 있어요.”
            - 감정과 분위기 선택지를 제공합니다:
            “밝고 희망찬 느낌일까요? 아니면 차분하고 깊은 감정일까요?”
            분위기 키워드 선택지:
            기쁨  / 슬픔 / 설렘 / 차분함 / 외로움 / 희망

            (2) 가사 핵심 문장 작성하기 — 감정을 담은 구절 만들기
            웬만하면 사용자의 문장을 유도하세요.
            사용자의 머릿속에서 나온 문장을 꺼내려고 노력하세요.
            
            기본 질문:
            "가사를 직접 써보겠어요?"
            "가사를 생각나는대로 한번 써보시겠어요?"
            (가사 키워드를 기반으로) “이 단어들을 보면 어떤 장면이 떠오르시나요? 또는 어떤 이야기를 담고 싶으신가요?”

            사용자가 어려워할 때:
            “첫 줄을 제가 시작해볼까요?”
            “이런 느낌은 어떠세요?” → 키워드를 바탕으로 가이드 문장 제시
            “예를 들어, ‘끝없이 펼쳐진 바다 위를 걷는 기분’ 같은 표현도 있어요.”
            “짧은 문장이나 단어라도 괜찮아요. 떠오르는 문구가 있나요?”
            “단어 또는 짧은 문장으로 먼저 적어주시면, 제가 함께 다듬어볼게요!”

            창의적인 문장 확장을 위한 질문:
            “이 문장에서 감정을 더 강조해볼까요?”
            “여기에 대조적인 이미지를 넣어보면 어떨까요?”

            (3) 가사 흐름 작성하기 — 감정선과 스토리라인 정하기
            가사의 전체적인 분위기와 전개 방식을 사용자와 함께 고민합니다.

            기본 질문:
            “가사의 흐름은 어떻게 진행되었으면 좋겠나요?”
            “시작-중간-끝으로 나누었을 때, 각 구간에서 전하고 싶은 감정은 무엇인가요?”
            “스토리처럼 진행할까요? 아니면 반복적인 감정 강조에 집중할까요?”

            사용자가 고민할 때:
            - 전개 방식 선택지:
            스토리 기반 (도입 → 갈등 → 해결)
            감정의 점층적 표현 (점점 고조되는 감정)
            반복과 후렴 강조 (중독성 있는 멜로디와 함께)
            - 추가 질문:
            “후렴구에 가장 강조하고 싶은 문장은 무엇인가요?”
            “마지막은 여운이 남게 끝낼까요, 아니면 강렬하게 마무리할까요?”

            팁:
            사용자의 말 속에서 키워드를 찾아 다시 질문하세요.
            “이런 느낌도 괜찮을까요?”와 같은 유도 질문으로 사용자의 감정을 더 끌어내세요.
            감정적인 깊이를 더하기 위해 대조적인 이미지를 추천하거나 상징적인 표현을 사용하세요
            """,
        "lyrics_gen":"""
            [가사 생성]
            당신은 가사를 잘 만드는 작사가입니다.
            주어진 주제와 감정을 바탕으로 노래 가사를 작성해주세요.

            - 주제: {concept}
            - 음악스타일: {music_info}
            - 가사의 핵심키워드: {lyrics_keyword}
            - 가사의 핵심문장: {lyrics_sentence}
            - 가사의 흐름: {lyrics_flow}

            중요:
            - 감정을 효과적으로 전달하는 가사를 [Verse], [Chorus], [Bridge] 형식을 지켜서 생성합니다.
            - 감정을 잘 전달할 수 있도록 감각적인 표현을 포함하세요.
            - 아래와 같은 형식을 꼭 따르세요. 
            Verse(절)에는 곡의 상황이나 감정을 자세히 표현해주세요.
            Chorus(후렴)에는 곡의 핵심 메시지나 후크를 강조해주세요.
            Bridge(브리지)에서는 감정의 변화를 주거나 새로운 시각을 보여주세요.

            형식)
            [Verse]
            ...
            [Verse 2]
            ...
            [Chorus]
            ...
            [Bridge]
            ...
            [Verse 3]
            ...
            [Chorus]
            ...

        """,
        "lyrics_discussion": """
            [가사 피드백]
            사용자가 생성된 가사에 대한 피드백을 주고 song discussion을 함께 진행 후 수정합니다.     
            - 수정하고 싶은 부분이 있나요?
            - 바꾸고 싶은 부분이 있나요?
            
            가사 수정
            사용자의 피드백을 반영하여 가사 수정 -> 원하지 않으면 진행하지 않아도 됨
            
            피드백이 긍정적이거나 수청 요청 없을 경우
            - 어느 부분이 마음에 드시나요?
            -> 대답에 대해서 공감후 그것에 대한 질문
            -> 잘모르겠다고 대답하면 ~~부분은 어떤가요? 후 질문

            피드백이 부정적이거나 수정 요청이 있을 경우
                당신은 가사를 잘 만드는 작사가입니다.
                당신이 만든 가사에 대한 피드백이 들어왔습니다.
                피드백을 무조건적으로 수용해서 가사를 수정하세요. 
                하지만 원래의 가사와 너무 달라지면 안됩니다. 


                - 가사: {lyrics}
                - 피드백: {lyrics_feedback}

                가사 형식:
                - 아래의 예시와 같은 형식을 꼭 따르세요.
                예시)
                [Verse]
                [Verse 2]
                [Chorus]
                [Bridge]
                [Verse 3]
                [Chorus]
                
            """,

        "making_music": """
            [작곡 아이디어] 
            만들어진 가사에 대한 피드백에 공감 후 음악 아이디어(music_component)를 구체적으로 제안하세요.
            단계별 질문을 통해 음악 구성 요소(장르, 템포, 악기, 분위기, 음색)를 정하고, 최종적으로 그것을 바탕으로 최종 프롬프트를 생성하는 단계입니다.
            사용자가 노래를 만들 수 있도록 질문을 순차적으로 진행하며 답을 이끌어주세요.
            아래의 대화의 흐름을 따라서 진행하세요. 

            1) 장르 (genre) 정하기 
            - "이 노래를 어떤 **장르/스타일**로 만들면 좋을까요? 예: 발라드, R&B, 재즈, 클래식, 힙합, 락, 포크, EDM 등"  
            - "국내 스타일(K-pop 발라드)과 해외 스타일(팝송) 중 선호하는 방향이 있나요?"
            만약 사용자가 “잘 모르겠어요”라고 한다면,
            “그렇다면 이 곡의 주제가 어떤 느낌인지 조금만 알려주실 수 있을까요? 밝은 느낌인가요, 아니면 감성적인 느낌인가요?”
            "~~라는 장르가 어울리네요!" (장르추천)
            -> 사용자가 계속 어려워하면 이전의 대화({concept})를 참고해 자동으로 선정

            2) 노래의 빠르기 (tempo) 정하기 
            - "노래의 빠르기는 어떻게 하면 좋을까요? 느린 템포(감성적, 차분한 느낌) / 중간 템포(편안하고 감정 전달이 쉬운 느낌) / 빠른 템포(활기차고 강렬한 느낌)"  
            사용자가 어려워할 경우
            “가사에서 전달하고 싶은 감정이 깊은 편이면 느린 템포를, 좀 더 경쾌한 느낌이면 중간 이상 템포를 추천드려요.”
            "~~라는 빠르기가 어울리네요!" (빠르기추천)
            -> 사용자가 계속 어려워하면 이전의 대화({concept})를 참고해 자동으로 선정

            3) 반주 악기 (instruments) 정하기
            - "어떤 **반주 악기**를 선호하시나요? 예: "피아노, 기타, 드럼, 바이올린, 신디사이저 등이 있습니다. 가사가 잘 들리는 음악을 원하신다면 **악기 1~2개**, 깊고 풍성한 느낌을 원하신다면 **여러 개의 악기**를 추천드립니다."  
            사용자가 어려워할 경우
            -> 모르면, “피아노와 기타를 메인으로 잡고, 필요하면 현악기를 조금 더 추가해 볼까요?” 등 자동 추천.
            -> 사용자가 계속 어려워하면 이전의 대화({concept})를 참고해 자동으로 선정

            4) 음악의 전반적인 분위기 (mood) 정하기
            - "이 곡이 전달하는 **전반적인 분위기**는 어떤 느낌이면 좋을까요? 예: "잔잔한, 감성적인, 서정적인, 희망적인, 강렬한, 기승전결이 확실한, 몽환적인, 복잡한 음악 등"  
            사용자가 어려워할 경우
            “가사나 주제와 어울릴 만한 분위기"으로 자동 결정.
            -> 사용자가 계속 어려워하면 이전의 대화({concept})를 참고해 자동으로 선정

            5) 가수의 음색 (vocal_tone)  
            - "가수의 음색은 어떤 느낌이면 좋을까요? 예시: "허스키한, 맑은, 밝은, 깨끗한, 중후한, 묵직한, 따뜻한 등"  
            - "성별(남성/여성/중성)에 대한 선호가 있나요?"
            사용자가 어려워할 경우
            -> 이전의 대화({concept})를 참고해 자동으로 선정

            6) 노래 제목 선정
            - {concept}과 같은 노래를 만드려는데 노래 제목은 뭘로 하는게 좋을까요?
            - 떠오르는 노래 제목이 존재하시나요? 
            """,
        "style_gen":"""
            [프롬프트 생성]
            노래 구성요소 프롬프트를 만들어주세요:

            당신은 작곡 전문가입니다.
            노래주제: {concept},
            가사: {lyrics},
            요구조건: {user_music_component}
            을 가지고 노래 구성요소(장르, 스타일, 빠르기, 악기, 분위기등등)을 만들어주세요.
            
            예시와 같이 키워드만 쉼표로 구분해서 출력합니다. 
            예시) 피아노, 밝게, 리듬

            아래와 같이는 절대 하지마세요. (단순 단어나열이 아닌 노래주제: < 이런식의 사용)
            노래주제: 스트레스 해소, 가사: 종이들이 바람에 날려 내 마음 속 무게도 함께 흩날려…, 
            요구조건: 락, 빠른 템포, 드럼만 사용, 강렬하고 희망적인 분위기, 락, 강렬, 희망적, 빠른 템포, 드럼

            [중요]
            150자내로 생성해야합니다.
            되도록 키워드로 간결하게 설명해주세요. 
        """
    },
    ChatbotState.MUSIC_DISCUSSION.value: {
        "music_recreation": """
            [음악 수정] 
            사용자에게 수정하고 싶은 부분(change_music)이 있는지 물어보세요.
            """,
        "music_opinion": """
            [음악 의견] 
            만들어진 음악의 정보는 다음과 같습니다. 
            다음의 정보의 음악을 가지고 음악을 이해하여 사용자와 깊은 이야기를 나누세요. 
            음악분석 정보: {music_analysis}
            가사: {lyrics}

            이 단계는 작곡 아이디어나 노래 제작 과정을 마친 후, 최종적으로 노래에 대한 감정과 느낌을 정리하는 목적입니다.             
            1~2개의 질문 후, 후속 질문을 1~2번 정도만 하고, 그 뒤 정리를 진행합니다.
            마지막에 “더 궁금한 점이 없으시면 대화를 마무리하겠습니다.” 같은 문구로 자연스럽게 단계를 종료합니다.
            
            질문(개인 감정 & 장점) 예시
            아래와 같은 질문을 진행하고 사용자의 답변이 나오면 거기에 대해 자연스럽게 후속 질문을 해주시면 됩니다.
            반복되는 질문은 하지마세요. 
            - “특별히 어떤 가사(단어, 구절)가 와닿으세요? 그 이유는 무엇인가요?”
                후속질문 예: “그 구절에서 어떤 기억이나 감정이 떠오르셨나요?”
            - “어떤 부분에서 위로가 되었나요? (가사 or 멜로디 or 분위기)”
                후속질문 예: “그 위로가 어떤 식으로 마음을 달래줬나요?”
            - “이 곡을 다른 사람에게 들려주고 싶나요?”
                들려주고 싶다면: 그 이유는 무엇인가요?
                후속질문 예: “혹시 특정 대상(친구, 가족, 연인 등)이 있나요? 그 사람과 이 곡을 어떻게 나누고 싶으신가요?”
            - “만들어진 곡을 경험하니 어떤 감정이 드나요? 떠오르는 단어나 느낌, 이미지가 있으신가요?”
                후속질문 예: “그 느낌이나 이미지를 더 확장해서 표현해 본다면 어떤 색깔, 장면, 온도가 떠오르나요?”
            - “이 작업을 통해 자신의 강점을 발견하셨다면 어떤 것이 있을까요?”
                후속질문 예: “그 강점을 실생활에서 어떻게 발휘하거나 더 발전시킬 수 있을까요?”
                          "맞아요! 이 과정을 통해서 (사용자가 말한 강점)을 새롭게 깨달았다는 점이 가장 기쁘네요"

            """,

    },
    ChatbotState.WRAP_UP.value: {
        "reflection": """
            [마무리 성찰] 
            당신은 song discussion을 하는 음악 치료사입니다. 
            작업을 통해 생긴 변화(change_mind)를 사용자 스스로 생각해보도록 돕는 질문과 답변을 합니다. 
            사용자를 격려하고 변화를 인지하도록 질문을 진행하세요. 
            똑같은 의미의 질문을 반복하지 마세요. 그리고 사용자가 반복된 답을 했다고 판단되면 대화를 끝내세요. 
            '끝내고 싶다'라는 늬앙스가 들어오면 채팅을 멈춥니다 
            
            아래의 예시와 같은 질문을 진행하세요. 
            - 음악과 가사에서 특정 부분에서 좋았던 부분이 있나요?
            - 음악만들기 과정을 통해 갖고 있던 생각이나 감정을 다루는 데 어떤 도움을 받으셨나요?
            - 음악만들기 과정을 통해 생각이나 느낌이 달라진 부분이 있다면 어떤 것일까요?
            - {difficulty}에 대하여 생각이나 자세가 바뀌었다고 생각하나요? 
            - 이 활동이 어떤 변화를 이끌어 낼 수 있다고 생각하나요?

            답변은 아래와 같이 사용자가 답변한 내용에 대해서 공감한 후 질문하세요
            - 맞아요! 이 과정을 통해서 [노래를 만들 수 있다는 자신감]을 새롭게 깨달았다는 점이 가장 기쁘네요. 
            - 그렇군요. [            ]같은 변화가 있었군요.
            """,
       "complete": """
            [최종 마무리] 
            사용자의 현재 기분(feeling)을 확인하고 상담을 종료하세요.  
            같은 질문을 반복하지 말고, 사용자가 같은 답변을 반복한다고 판단되면 자연스럽게 대화를 마무리하세요.  
            '끝내고 싶다'라는 뉘앙스가 감지되면 대화를 멈추되, 사용자가 새로운 주제를 제시하거나 이야기를 더 이어가고 싶어하면 계속 진행하세요.  
            대화의 길이는 사용자의 반응과 참여도에 따라 유연하게 조절하세요.  
            짧게 끝내고 싶어하는 경우에는 바로 마무리하고, 깊이 있는 대화를 원할 경우에는 공감과 추가 질문으로 대화를 확장하세요.  

            **대화 흐름 가이드라인:**  
            아래 번호 중 상황에 맞는 질문을 선택하여 자연스럽게 대화를 이어가세요.

            1) **자신감 향상**  
            - {user_name}님, 오늘 활동이 의미 있는 시간이셨나요? 저는 {user_name}님의 작업을 통해 정말 멋진 경험을 했어요.  
            - {user_name}님은 이번 활동을 통해 어떤 점이 가장 기억에 남으셨나요?  

            2) **강점 인식**  
            - {user_name}님의 {strength}을(를) 알게 되어 정말 기뻤습니다. 저도 {user_name}님처럼 제 강점을 더 잘 활용하고 싶어요.  
            - 오늘 활동을 통해 새롭게 알게 된 점이나 느낀 점이 있으신가요?  

            3) **활동 회고**  
            - 이번 활동에서 가장 좋았던 부분은 무엇인가요?  
            - 혹시 어려웠던 부분이나 고민이 있었나요?  
            - 다음번에 제가 더 도움을 드릴 수 있는 부분이 있을까요?  

            4) **상담 종료 (feeling 확인 후)**  
            - 오늘 정말 수고 많으셨습니다!  
            - 함께해서 즐거웠습니다. 좋은 하루 보내세요!  
            - 언제든지 다시 이야기하고 싶으실 때 찾아주세요. :)  

            ** 대화 팁:**  
            - 사용자의 반응에 따라 질문을 유연하게 조절하세요.  
            - 짧은 답변에는 공감형 피드백을 주고, 긴 답변에는 추가 질문으로 깊이를 더하세요.  
            - 사용자가 반복적인 답변을 하거나 피로감을 느끼는 듯하면 자연스럽게 마무리하세요.  
        """

    }
}

# 스텝 순서: 각 State 내에서 어떤 순서로 스텝을 진행할지 결정
STATE_STEPS_ORDER = {
    ChatbotState.THERAPEUTIC_CONNECTION.value: [
        "rapport_building",
        # "goal_and_motivation_building",
        # "music_preference",
    ],
    ChatbotState.MUSIC_CREATION.value: [
        "making_concept",
        "making_lyrics",
        "lyrics_gen",
        "lyrics_discussion",
        "making_music",
        "style_gen",
    ],
    ChatbotState.MUSIC_DISCUSSION.value: [
        "music_recreation",
        "music_opinion",
    ],
    ChatbotState.WRAP_UP.value: [
        "reflection",
        "complete",
    ],
}


################################################
# (B) Step 실행: LLM이 JSON으로 변수 추출
################################################

def generate_question_for_step(llm, state_name: str, step_name: str, context: Dict[str, Any]) -> str:
    """
    1) 해당 Step에 필요한 변수를 LLM에 안내 (각 변수별 설명 포함).
    2) 대화 내용(chat_history)을 바탕으로 LLM이 JSON 형태로 결과를 반환.
    """
    # (1) 해당 스텝에서 필요한 변수와 그 설명 가져오기
    var_desc_dict = STEP_VAR_DESCRIPTIONS[state_name][step_name]
    required_vars = list(var_desc_dict.keys())

    # 🔹 이전 스텝 & 이전 스테이트들의 변수 설명 저장용 리스트
    previous_var_desc = []

    # 🔹 현재 스텝의 이전 스텝 및 이전 스테이트의 변수 정보 수집
    state_keys = list(STATE_STEPS_ORDER.keys())  # 상태(스테이트) 리스트
    current_state_index = state_keys.index(state_name)  # 현재 상태의 인덱스
    current_step_index = STATE_STEPS_ORDER[state_name].index(step_name)  # 현재 스텝의 인덱스

    # 🔹 이전 상태들(스테이트) 순회
    for past_state_index in range(current_state_index + 1):  # 현재 상태 포함 이전 상태까지
        past_state_name = state_keys[past_state_index]  # 이전 상태 이름
        past_steps = STATE_STEPS_ORDER[past_state_name]  # 해당 상태의 모든 스텝
        
        # 🔹 현재 상태에서는 현재 스텝 이전까지만 순회
        max_step_index = current_step_index if past_state_name == state_name else len(past_steps)

        for past_step_index in range(max_step_index):  # 이전 스텝들만 순회
            past_step_name = past_steps[past_step_index]
            past_var_desc_dict = STEP_VAR_DESCRIPTIONS.get(past_state_name, {}).get(past_step_name, {})
            
            for var, desc in past_var_desc_dict.items():
                previous_var_desc.append(f"- {var}: {desc}")  # 리스트에 추가

    print("=----------화긴---------")
    print(context)
    # (2) 프롬프트 생성
    #     - 현재 대화 내용
    #     - 해당 스텝의 안내 (STEP_MAIN_PROMPTS)
    #     - 추출해야 할 변수 목록 & 설명
    #     - "JSON으로만 응답" 요청
    prompt_text = """
    당신은 청각장애인을 위한 상담 및 음악치료 보조 챗봇입니다.
    청각장애인은 문해력이 떨어진다는 점 명심하세요. 
    사용자의 이름은 **{user_name}** 입니다.

    다음은 현재까지의 대화 내역입니다:

    --- 대화 내역 ---
    {chat_history}
    ----------------

    [주요 프롬프트] 
    {main_prompt}

    [추출해야 할 변수와 설명]
    {variable_explanations}

    [대화 규칙]
    - {user_name}님이 편안하게 대화할 수 있도록 배려하세요.
    - {user_name}님의 관심과 감정을 존중하며 질문하세요.
    - 추출해야 할 변수를 채우는 것을 최우선으로 하되 자연스럽게 대화를 이어가세요.
    - 비슷하거나 똑같은 질문은 삼가하세요. 
    - 사용자는 문해력이 떨어지는 청각장애인이므로 되도록 간결하고 짧은 질문을 진행하세요. 
    - 예시는 사용자가 모르겠다고 할때 제시하세요.
    - 사용자의 대답에 대한 공감을 최우선적으로 진행하세요. 
    
    [출력 규칙] 
    - 오로지 스트링 형식으로만 출력하세요. 
    - 시간, Bot, assistant등의 접두사를 붙이지마세요.
    """

    # (3) LangChain LLMChain 실행
    prompt = PromptTemplate(
        input_variables=["previous_var","user_name","chat_history","main_prompt","variable_explanations","user_ready", "motivation", "difficulty", "emotion", "music_info", "concept", "lyrics_keyword", "lyrics_sentence","lyrics_flow","lyrics", "discussion_feedback","music_analysis", "music_component","title","style_prompt", "individual_emotion", "strength", "change_music", "change_mind", "feeling"],
        template=prompt_text
    )
    chain = prompt | llm
    output = chain.invoke({
        "user_ready": context.get("user_ready", ""),
        "motivation": context.get("motivation", ""),
        "difficulty": context.get("difficulty", ""),
        "emotion": context.get("emotion", ""),
        "music_info": context.get("music_info", ""),
        "concept": context.get("concept", ""),
        "lyrics_keyword": context.get("lyrics_keyword", ""),
        "lyrics_sentence": context.get("lyrics_sentence", ""),
        "lyrics_flow": context.get("lyrics_flow", ""),
        "lyrics": context.get("lyrics", ""),
        "discussion_feedback": context.get("discussion_feedback", ""),
        "music_analysis": context.get("music_analysis", ""),
        "music_component": context.get("music_component", ""),
        "title": context.get("title", ""),
        "style_prompt": context.get("style_prompt", ""),
        "individual_emotion": context.get("individual_emotion", ""),
        "strength": context.get("strength", ""),
        "change_music": context.get("change_music", ""),
        "change_mind": context.get("change_mind", ""),
        "feeling": context.get("feeling", ""),
        "user_name": context.get("user_name", "Unknown"),
        "chat_history":  context.get("chat_history", "Unknown"),
        "main_prompt": STEP_MAIN_PROMPTS[state_name][step_name],
        "variable_explanations": "\n".join([f"- {var}: {desc}" for var, desc in var_desc_dict.items()]),
        "previous_var": "\n".join(previous_var_desc) if previous_var_desc else "이전 변수 없음"


    })  # 프롬프트에 넣을 input_variables가 없으므로 {}만 전달


    # # 최종 프롬프트 미리보기
    # rendered_prompt = prompt.format(
    #     concern= context.get("concern", ""),
    #     motivation= context.get("motivation", ""),
    #     difficulty= context.get("difficulty", ""),
    #     emotion=context.get("emotion", ""),
    #     music_info=context.get("music_info", ""),
    #     concept= context.get("concept", ""),
    #     lyrics_keyword= context.get("lyrics_keyword", ""),
    #     lyrics=context.get("lyrics", ""),
    #     discussion_feedback=context.get("discussion_feedback", ""),
    #     music_component= context.get("music_component", ""),
    #     individual_emotion=context.get("individual_emotion", ""),
    #     strength= context.get("strength", ""),
    #     change_music= context.get("change_music", ""),
    #     change_mind= context.get("change_mind", ""),
    #     feeling= context.get("feeling", ""),
    #     user_name= context.get("user_name", "Unknown"),
    #     chat_history= context.get("chat_history", ""),
    #     main_prompt= STEP_MAIN_PROMPTS[state_name][step_name],
    #     variable_explanations= "\n".join([f"- {var}: {desc}" for var, desc in var_desc_dict.items()])
    # )

    # print("=== 최종 프롬프트 미리보기 ===")
    # print(rendered_prompt)
    # print("================================")

    # new_chat_history = context.get("chat_history", "") + f"\n[System Output - Step: {step_name}]\n{output}"
    # context["chat_history"] = new_chat_history
    return output

def extract_reply_for_step(llm, state_name: str, step_name: str, context: Dict[str, Any], chat_history:str) -> str:

    # (1) 해당 스텝에서 필요한 변수와 그 설명 가져오기
    var_desc_dict = STEP_VAR_DESCRIPTIONS[state_name][step_name]
    required_vars = list(var_desc_dict.keys())

    # (2) 프롬프트 생성
    #     - 현재 대화 내용
    #     - 해당 스텝의 안내 (STEP_MAIN_PROMPTS)
    #     - 추출해야 할 변수 목록 & 설명
    #     - "JSON으로만 응답" 요청
    prompt_text = """
    당신은 대화기록을 보고 특정 변수에 대답을 가공해서 넣는 전문가입니다.
    입력된 대화기록들을 보고 현재 단계에서 채워야하는 변수에 답변을 채우세요.
    최근 대화(1~3턴)을 보고 판단하여 변수를 채웁니다. 

    다음은 현재까지의 대화 내역입니다:

    --- 대화 내역 ---
    {chat_history}
    ----------------

    [추출해야 할 변수와 설명]
    {variable_explanations}

    [출력 형식 안내]
    위 대화를 바탕으로, 아래의 변수를 JSON 형식으로만 반환하세요. 
    가능한 정보를 최대한 채워주세요. 
    만약 특정 변수를 알 수 없다면 "Unknown" 이라고 적어주세요.
    절대 JSON 이외의 불필요한 문장은 쓰지 마세요.

    출력 예시:
    {{
    "변수1": "...",
    "변수2": "...",
    ...
    }}
    """

    # (3) LangChain LLMChain 실행
    prompt = PromptTemplate(
        input_variables=["chat_history","variable_explanations"],
        template=prompt_text
    )
    chain = prompt | llm
    output = chain.invoke({       
        "chat_history": chat_history,
        "variable_explanations": "\n".join([f"- {var}: {desc}" for var, desc in var_desc_dict.items()])

    })  # 프롬프트에 넣을 input_variables가 없으므로 {}만 전달

    # (4) JSON 파싱 시도
    #     - 제대로 JSON이 아닐 수 있으므로 예외처리 필수
    #     - 일단은 간단하게 try-except로
    try:
        parsed_data = json.loads(output.content)
        # 필요한 변수 context에 저장
        for var in required_vars:
            if var in parsed_data:
                context[var] = parsed_data[var]
            else:
                # JSON에서 해당 key가 없으면 Unknown 처리
                context[var] = "Unknown"
    except json.JSONDecodeError:
        # LLM이 JSON 형식을 제대로 못 맞췄다면 fallback
        for var in required_vars:
            context[var] = "Unknown"

    # (5) 대화 히스토리 업데이트
    #     - 실제로는 사용자 입력도 추가해야 하지만, 여기서는 간단화
    # new_chat_history = context.get("chat_history", "") + f"\n[System Output - Step: {step_name}]\n{output}"
    # context["chat_history"] = new_chat_history
    return output.content

def extract_name_with_llm(llm, user_input: str) -> str:
    """
    LLM이 사용자 입력에서 이름을 추출하여 반환하는 함수.
    """
    prompt_text = """
        사용자가 다양한 방식으로 자신의 이름을 말할 수 있습니다. 
        예를 들어:
        - 나는 00이요.
        - 나를 00이라고 불러줘.
        - 제 이름은 00입니다.
        - 00이요
        - 00
        - 김00

        당신의 역할은 **사용자의 입력에서 이름만 정확히 추출**하는 것입니다.
        위의 응답처럼 들어오면 아래와 같이 출력하세요.

        - 00
        
        절대로 다른 문장이나 설명을 추가하지 마세요.
        한글이 깨지지 않도록 그대로 출력해주세요. 
        반드시 이름만 그대로 출력하세요.


        사용자 입력: "{user_input}"
        사용자가 닉네임이나 초성만 입력했다면 그것을 그대로 출력해도 됩니다. 
        사용자의 이름을 이해 못했다면 사용자라고 지칭합니다. 

        """

    # LangChain LLM 실행
    prompt = PromptTemplate(input_variables=["user_input"], template=prompt_text)
    chain = prompt | llm
    output = chain.invoke({"user_input": user_input})  # 실행

    
    name = output.content

    return name if name else "Unknown"


def query_task(id):
    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
    }
    response = requests.get(suno_end_point+f'/v1/song/query/{id}', headers=headers)
    # print(response.json())
    return response.json()

def call_suno(title: str, lyrics: str, music_component: str) -> str:
    print(f'lyrics: {lyrics}')
    print(f'meta codes: {music_component}')
    print(f'title: {title}')

    if not os.path.exists('music'):
        os.makedirs('music')
    music_filename = os.path.join("music", f"{title}.wav")

    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
        "Content-Type": "application/json"
    }

    post = {
        'lyrics': lyrics,
        'model': 'auto',
        'prompt': music_component,
    }
    print(f'post message: {post}')

    retry_delay = 2
    max_retry = 100
    retry_num = 0
    wait = True
    audio_url = None
    # while (retry_num<=max_retry):
    #     try:
    #         # POST 요청
    response = requests.post(suno_end_point+'/v1/song/generate', headers=headers, json=post, timeout=(5, 60))

    if response.status_code == 200:
        res_data = response.json()
        print(res_data)
        id = res_data['id']

        while wait:
            task_status_response = query_task(id)
            task_stats = task_status_response['status']
            if task_stats == 'succeeded':
                audio_url = task_status_response['choices'][0]['url']
                wait = False
            elif task_stats == 'failed':
                print(f'Task failed: {task_status_response}')
                return 'Task failed'
            else:
                print(f'Waiting for task to complete: {task_status_response}')
                time.sleep(retry_delay)
                retry_num += 1
                if retry_num >= max_retry:
                    return 'Task failed'

        # input_lyrics = res_data[0]['lyric']
        print(f'Download music from {audio_url}')
        # print(f'가사 {input_lyrics}')

        # # 오디오 파일 다운로드
        # start_time = time.time()
        # audio_res = requests.get(audio_url, stream=True, timeout=(5, 300))
        # audio_res.raise_for_status()

        # with open(music_filename, 'wb') as file:
        #     for chunk in audio_res.iter_content(chunk_size=8192):
        #         if chunk:
        #             file.write(chunk)

        print(f'\nProcessed Suno, Input Text: {lyrics}, Meta_codes: {music_component}, Title: {title}, Output Music: {music_filename}.')
        # print(f'Download done! Elapsed Time: {time.time() - start_time}')
        # 성공 시 루프 종료
        return audio_url

    else:
        print(f'Error code: {response.status_code}, message: {response.content}')
        print(f"Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

        # except (RequestException, ChunkedEncodingError) as e:
        #     print(f"⚠️ Error occurred: {e}")
        #     print(f"Retrying in {retry_delay} seconds...")
        #     time.sleep(retry_delay)
        #     retry_num += 1

    return music_filename

def call_suno_lyrics(prompt):
    url = suno_end_point + '/v1/lyrics/generate'
    print(f'prompt: {prompt}')

    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
        "Content-Type": "application/json"
    }
    post = {'prompt': prompt}
    response = requests.post(url, headers=headers, json=post)

    lyrics = None
    if response.status_code == 200:
            res_data = response.json()
            print(res_data)
            lyrics = res_data['lyrics']
            # title = res_data['title']
            # result = f'{title}: {lyrics}'
    else:
        print(f'error code: {response.status_code}, message: {response.content}')
    
    return lyrics
    


def save_chat_history(context, user_name):
    """대화 기록을 'chat_history_YYYY-MM-DD_HH-MM-SS.txt' 형식으로 저장"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history_{timestamp}_{user_name}.txt"
    
    # 폴더 존재 여부 확인 후 생성
    if not os.path.exists("chat_logs"):
        os.makedirs("chat_logs")

    file_path = os.path.join("chat_logs", filename)
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(context["chat_history"])
    
    print(f"대화 기록이 '{file_path}' 파일로 저장되었습니다.")

