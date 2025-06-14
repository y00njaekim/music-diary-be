from llm_instance import llm
from .therapeutic_connection import therapeutic_connection
from .lyrics_creation import extraction_source, making_lyrics
from .music_creation import music_making, music_creation
from .music_discussion import music_discussion
from database.manager import DBManager, SEARCH_OPTION


from typing import TypedDict, Tuple, Union
from langchain_core.memory import BaseMemory
import json
from enum import Enum


class TherapeuticConnectionSlot(TypedDict):
    name: str
    therapy_difficulty: str
    difficulty: str
    difficulty_category: str
    motivation: str


class ExtractionSourceSlot(TypedDict):
    concept: str
    concept_discussion: str
    lyric_keyword: str
    lyric_image: str
    lyrics_content: str


class MakingLyricsSlot(TypedDict):
    lyrics: str


class MusicMakingSlot(TypedDict):
    music_information: str
    genre: str
    instrument: str
    mood: str
    vocal: str
    tempo: str


class MusicDiscussionSlot(TypedDict):
    individual_emotion: str
    change_mind: str
    change_attitude: str
    touching_lyrics: str
    strength: str
    feeling: str


class CombinedSlot(TypedDict, total=False):
    # therapeutic_connection
    name: str
    therapy_difficulty: str
    difficulty: str
    difficulty_category: str
    motivation: str
    # extraction_source
    concept: str
    concept_discussion: str
    lyric_keyword: str
    lyric_image: str
    lyrics_content: str
    # making_lyrics
    lyrics: str
    # music_making
    music_information: str
    genre: str
    instrument: str
    mood: str
    vocal: str
    tempo: str
    # music_discussion
    individual_emotion: str
    change_mind: str
    change_attitude: str
    touching_lyrics: str
    strength: str
    feeling: str


class State(Enum):
    THERAPEUTIC_CONNECTION = "therapeutic_connection"
    EXTRACTION_SOURCE = "extraction_source"
    MAKING_LYRICS = "making_lyrics"
    MUSIC_MAKING = "music_making"
    MUSIC_CREATION = "music_creation"
    MUSIC_DISCUSSION = "music_discussion"


STATE_NEXT = {
    State.THERAPEUTIC_CONNECTION: State.EXTRACTION_SOURCE,
    State.EXTRACTION_SOURCE: State.MAKING_LYRICS,
    State.MAKING_LYRICS: State.MUSIC_MAKING,
    State.MUSIC_MAKING: State.MUSIC_CREATION,
    State.MUSIC_CREATION: State.MUSIC_DISCUSSION,
    State.MUSIC_DISCUSSION: None,  # 마지막 상태
}


def execute_state(
    user_input: str, state: State, turn: int, slot: CombinedSlot, memory: BaseMemory, summary: BaseMemory, db_manager: DBManager,
) -> Tuple[str, int, CombinedSlot]:
    flag = 0

    # state지정
    if state == State.THERAPEUTIC_CONNECTION:
        func = therapeutic_connection
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

    # 답변 생성
    if turn == 0:
        response, state_slot = func(json.dumps(slot, ensure_ascii=False), llm, memory)
        # print(response)
    else:
        response, state_slot = func(user_input, llm, memory)
        # print(response)

    none_fields = 0
    for k, v in state_slot.model_dump().items():
        if k not in slot:
            slot[k] = v
        else:
            if v is not None:
                slot[k] = v

        if v is None:
            none_fields += 1

    sid = db_manager.get_current_session()
    _ = db_manager.insert_chat(sid, user_input, True)
    _ = db_manager.insert_chat(sid, response, False)
    _ = db_manager.insert_keywords(sid, slot)

    if state == State.MAKING_LYRICS:
        flag = 1
        lyrics = slot["lyrics"]
        _ = db_manager.insert_lyrics(sid, lyrics)

    # TODO: 노래 생성에 실패해도 그냥 다음 state로 넘어가???
    if state == State.MUSIC_CREATION:
        flag = 1
        style_elems = []
        for k in ['genre', 'instrument', 'mood', 'vocal', 'tempo']:
            if slot[k] is not None:
                style_elems.append(slot[k])
        style = ", ".join(style_elems)
        # title = slot['name']  # probably it is a username

        # TODO: 항상 제일 마지막으로 저장된 lyrics가 음악 생성에 사용될 가사라는 가정이 깔리있음. 아닌 경우는 없는 지 확인 필요.
        lyrics_id = db_manager.search("lyrics", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["lyrics_id"]
        url = response.split(":")[-1].strip()
        _ = db_manager.insert_music(sid, lyrics_id, style, url, "").data[0]["music_id"]

    # 다음 state로 넘어갈지 flag
    if none_fields == 0:
        # 버튼뜨는 타이밍
        print("all slot filled")
        flag = 1

    if turn > 1: # TODO: check 1 -> 5?
        print("over the 5 turn")
        flag = 1

    return response, flag, slot
