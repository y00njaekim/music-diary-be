from langchain_openai import ChatOpenAI
from .therapeutic_connection import therapeutic_connection
from .lyrics_creation import extraction_source, making_lyrics
from .music_creation import music_making
from .music_discussion import music_discussion


from typing import TypedDict, Tuple, Union
from langchain_core.memory import BaseMemory

llm = ChatOpenAI(model="gpt-4.1", temperature=0)


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
    # therpeutic_connection
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


def execute_state(
    user_input: str, state: str, turn: int, slot: CombinedSlot, memory: BaseMemory
) -> Tuple[str, int, Union[TherapeuticConnectionSlot | ExtractionSourceSlot | MakingLyricsSlot | MusicMakingSlot | MusicDiscussionSlot]]:
    flag = 0

    # state지정
    if state == "therpeutic_connection":
        func = therapeutic_connection
    elif state == "extraction_source":
        func = extraction_source
    elif state == "making_lyrics":
        func = making_lyrics
    elif state == "music_making":
        func = music_making
    elif state == "music_discussion":
        func = music_discussion

    # 답변 생성
    if turn != 0:
        response, state_slot = func(user_input, llm, memory)
        print(response)

    else:
        response, state_slot = func(slot, llm, memory)
        print(response)

    # slot 다 채웠는지 확인
    none_fields = {k: v for k, v in state_slot.model_dump().items() if v is None}

    # 다음 state로 넘어갈지 flag
    if len(none_fields) == 0:
        # 버튼뜨는 타이밍
        print("all slot filled")
        flag = 1

        return response, flag, state_slot.model_dump()

    if state == "making_lyrics":
        flag = 1
        return response, flag, state_slot.model_dump()

    if turn > 20:
        print("over the 20 turn")
        flag = 1
        return response, flag, state_slot.model_dump()

    return response, flag, state_slot.model_dump()
