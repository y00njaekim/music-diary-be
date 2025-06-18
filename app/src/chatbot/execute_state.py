from llm_instance import llm
from .therapeutic_connection import therapeutic_connection
from .therapeutic_connection_re import therapeutic_connection_re
from .lyrics_creation import extraction_source, making_lyrics
from .music_creation import music_making, music_creation
from .music_discussion import music_discussion
from .termination import termination
from database.manager import DBManager, SEARCH_OPTION
from .types import CombinedSlot, State
import json
from typing import Tuple
from langchain_core.memory import BaseMemory
from flask import request

STATE_NEXT = {
    State.THERAPEUTIC_CONNECTION: State.EXTRACTION_SOURCE,
    State.EXTRACTION_SOURCE: State.MAKING_LYRICS,
    State.MAKING_LYRICS: State.MUSIC_MAKING,
    State.MUSIC_MAKING: State.MUSIC_CREATION,
    State.MUSIC_CREATION: State.MUSIC_DISCUSSION,
    State.MUSIC_DISCUSSION: State.TERMINATION,
    State.TERMINATION: None,  # 마지막 상태
}


def execute_state(
    user_input: str,
    state: State,
    turn: int,
    slot: CombinedSlot,
    memory: BaseMemory,
    summary: str,
    db_manager: DBManager,
) -> Tuple[str, int, CombinedSlot]:
    flag = 0

    # state지정
    if state == State.THERAPEUTIC_CONNECTION:
        if summary is None:
            func = therapeutic_connection
        else:
            func = therapeutic_connection_re
    elif state == State.EXTRACTION_SOURCE:
        func = extraction_source
    elif state == State.MAKING_LYRICS:
        func = making_lyrics
    elif state == State.MUSIC_MAKING:
        func = music_making
    elif state == State.MUSIC_CREATION:
        func = music_creation
    elif state == State.MUSIC_DISCUSSION:
        func = music_discussion
    elif state == State.TERMINATION:
        func = termination

    # JWT에서 추출한 사용자 정보
    jwt_user = request.jwt_user
    user_id = jwt_user["id"]

    # 쿼리 파라미터에서 sid 추출 및 출력
    front_sid = request.args.get("sid")
    sid = db_manager.search("diary", "user_id", user_id, SEARCH_OPTION.ID.value, id=front_sid)
    sid = sid.data[0]["session_id"]

    if state == State.TERMINATION:
        flag = 0

        chat_records_response = db_manager.search("chat", "session_id", sid, SEARCH_OPTION.ALL).data
        dialogue = ""
        # 검색 결과가 있는지 확인합니다.
        if chat_records_response:
            # 각 채팅 레코드(딕셔너리)를 순회합니다.
            for record in chat_records_response:
                # 각 레코드의 'text' 필드 값을 가져와 dialogue에 이어 붙입니다.
                # record['text']는 해당 채팅 메시지의 내용을 담고 있습니다.
                dialogue += record["text"] + "\n"  # 각 메시지 뒤에 줄바꿈을 추가하여 구분

            # 마지막에 추가된 불필요한 줄바꿈을 제거할 수도 있습니다.
            dialogue = dialogue.strip()

        else:
            # 검색된 채팅 레코드가 없는 경우
            print(f"세션 ID '{sid}'에 대한 채팅 기록이 없습니다.")
            dialogue = ""  # 대화 기록이 없으면 빈 문자열로 설정

        response, summary = func(dialogue, llm, slot)

        # summary db 저장
        latest_chat = db_manager.search("chat", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["chat_id"]
        latest_keywords = db_manager.search("keywords", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["keywords_id"]
        latest_lyrics = db_manager.search("lyrics", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["lyrics_id"]
        latest_music = db_manager.search("music", "lyrics_id", latest_lyrics, SEARCH_OPTION.LATEST.value).data[0]["music_id"]
        latest_state = db_manager.search("state", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["state_id"]

        db_manager.insert_summary(sid, summary, latest_chat, latest_music, latest_state, latest_keywords)

        return response, flag, slot

    # 답변 생성
    if turn == 0:
        response, state_slot = func(json.dumps(slot, ensure_ascii=False), llm, memory, slot)
    else:
        response, state_slot = func(user_input, llm, memory, slot)

    none_fields = 0
    for k, v in state_slot.model_dump().items():
        if k not in slot:
            slot[k] = v
        else:
            if v is not None:
                slot[k] = v

        if v is None:
            none_fields += 1

    chat_res = db_manager.insert_chat(sid, user_input, True)
    chat_id = chat_res.data[0]["chat_id"]
    _ = db_manager.insert_chat(sid, response, False)
    _ = db_manager.insert_keywords(sid, slot)

    if state == State.MAKING_LYRICS:
        flag = 1

        lyrics = state_slot.lyrics
        print(lyrics)
        _ = db_manager.insert_lyrics(sid, chat_id, lyrics)

    # TODO: 노래 생성에 실패해도 그냥 다음 state로 넘어가???
    if state == State.MUSIC_CREATION:
        flag = 1
        style_elems = []
        for k in ["genre", "instrument", "mood", "vocal", "tempo"]:
            if slot[k] is not None:
                style_elems.append(slot[k])
        style = ", ".join(style_elems)
        # title = slot['name']  # probably it is a username

        # TODO: 항상 제일 마지막으로 저장된 lyrics가 음악 생성에 사용될 가사라는 가정이 깔리있음. 아닌 경우는 없는 지 확인 필요.
        # => 제일 마지막으로 저장된 lyrics을 무조건적으로 사용한다는 시나리오로 가야할 것 같습니다.
        lyrics_id = db_manager.search("lyrics", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["lyrics_id"]
        url = response.split(":")[-1].strip()
        _ = db_manager.insert_music(lyrics_id, style, url, "").data[0]["music_id"]

    # 다음 state로 넘어갈지 flag
    if none_fields == 0:
        # 버튼뜨는 타이밍
        print("all slot filled")
        flag = 1

    if turn > 1:  # TODO: check 1 -> 5?

        print("over the 5 turn")
        flag = 1

    return response, flag, slot
