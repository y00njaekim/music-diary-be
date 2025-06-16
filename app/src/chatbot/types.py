from typing import TypedDict
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
