from llm_instance import llm
from .therapeutic_connection import therapeutic_connection
from .lyrics_creation import extraction_source, making_lyrics
from .music_creation import music_making
from .music_discussion import music_discussion


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
    MUSIC_DISCUSSION = "music_discussion"


STATE_NEXT = {
    State.THERAPEUTIC_CONNECTION: State.EXTRACTION_SOURCE,
    State.EXTRACTION_SOURCE: State.MAKING_LYRICS,
    State.MAKING_LYRICS: State.MUSIC_MAKING,
    State.MUSIC_MAKING: State.MUSIC_DISCUSSION,
    State.MUSIC_DISCUSSION: None,  # 마지막 상태
}


def execute_state(
    user_input: str, state: State, turn: int, slot: CombinedSlot, memory: BaseMemory
) -> Tuple[str, int, Union[TherapeuticConnectionSlot | ExtractionSourceSlot | MakingLyricsSlot | MusicMakingSlot | MusicDiscussionSlot]]:
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
    elif state == State.MUSIC_DISCUSSION:
        func = music_discussion

    # 답변 생성
    if turn != 0:
        response, state_slot = func(user_input, llm, memory)
        print(response)

    else:
        response, state_slot = func(json.dumps(slot, ensure_ascii=False), llm, memory)
        print(response)

    # slot 다 채웠는지 확인
    none_fields = {k: v for k, v in state_slot.model_dump().items() if v is None}

    # 다음 state로 넘어갈지 flag
    if len(none_fields) == 0:
        # 버튼뜨는 타이밍
        print("all slot filled")
        flag = 1

        return response, flag, state_slot.model_dump()

    if state == State.MAKING_LYRICS:
        flag = 1
        return response, flag, state_slot.model_dump()

    if turn > 5:
        print("over the 5 turn")
        flag = 1
        return response, flag, state_slot.model_dump()

    return response, flag, state_slot.model_dump()
