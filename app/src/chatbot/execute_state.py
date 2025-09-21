from llm_instance import llm
from .therapeutic_connection import therapeutic_connection
from .therapeutic_connection_re import therapeutic_connection_re
from .lyrics_creation import extraction_source, making_lyrics
from .music_creation import music_making, music_creation
from .music_discussion import music_discussion
from .termination import termination
from database.manager import DBManager, SEARCH_OPTION
from .types import CombinedSlot, State, MusicMakingSlot, MusicDiscussionSlot
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
    State.TERMINATION: None,  # final state
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

    # Select the state handler
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

    # User information extracted from the JWT
    jwt_user = request.jwt_user
    user_id = jwt_user["id"]

    # Retrieve sid from the query parameter
    front_sid = request.args.get("sid")
    sid = db_manager.search("diary", "user_id", user_id, SEARCH_OPTION.ID.value, id=front_sid)
    sid = sid.data[0]["session_id"]


    # Reset slots when re-entering the music-making phase
    if state==State.MUSIC_MAKING:
        if "individual_emotion" in slot.keys():
            music_making_keys=MusicMakingSlot.__annotations__.keys()
            music_discussion_keys=MusicDiscussionSlot.__annotations__.keys()

            for k in music_making_keys:
                slot[k] = None
            for k in music_discussion_keys:
                slot[k] = None

    if state == State.TERMINATION:
        flag = 0

        chat_records_response = db_manager.search("chat", "session_id", sid, SEARCH_OPTION.ALL).data
        dialogue = ""
        # Check whether any chat records were found
        if chat_records_response:
            # Iterate through each chat record dictionary
            for record in chat_records_response:
                # Append each record's text field to build the dialogue string
                dialogue += record["text"] + "\n"

            # Remove the trailing newline that was added
            dialogue = dialogue.strip()

        else:
            # No chat records were found
            print(f"No chat history found for session '{sid}'.")
            dialogue = ""  # Use an empty string when there is no dialogue

        response, summary = func(dialogue, llm, slot)

        # Save the summary to the database
        latest_chat = db_manager.search("chat", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["chat_id"]
        latest_keywords = db_manager.search("keywords", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["keywords_id"]
        latest_lyrics = db_manager.search("lyrics", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["lyrics_id"]
        latest_music = db_manager.search("music", "lyrics_id", latest_lyrics, SEARCH_OPTION.LATEST.value).data[0]["music_id"]
        latest_state = db_manager.search("state", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["state_id"]

        db_manager.insert_summary(sid, summary, latest_chat, latest_music, latest_state, latest_keywords)

        return response, flag, slot

    # Generate a response
    if turn == 0:
        if (state == State.THERAPEUTIC_CONNECTION) and (summary != None):
            response, state_slot = func(json.dumps(slot, ensure_ascii=False), summary, llm, memory, slot)
        else:
            response, state_slot = func(json.dumps(slot, ensure_ascii=False), llm, memory, slot)
    else:
        if (state == State.THERAPEUTIC_CONNECTION) and (summary != None):
            response, state_slot = func(user_input, summary, llm, memory, slot)
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

    # TODO: Should we advance to the next state even if song generation fails?
    if state == State.MUSIC_CREATION:
        flag = 1
        style_elems = []
        for k in ["genre", "instrument", "mood", "vocal", "tempo"]:
            if slot[k] is not None:
                style_elems.append(slot[k])
        style = ", ".join(style_elems)
        # title = slot['name']  # probably it is a username

        # TODO: We assume the most recently saved lyrics are always used for music generation. Validate this assumption.
        lyrics_id = db_manager.search("lyrics", "session_id", sid, SEARCH_OPTION.LATEST.value).data[0]["lyrics_id"]
        url = response.split(":")[-1].strip()
        _ = db_manager.insert_music(lyrics_id, style, url, "").data[0]["music_id"]

    # Decide whether to advance to the next state
    if none_fields == 0:
        # Trigger flow control when every slot is filled
        print("all slot filled")
        flag = 1

    if turn > 6:  # TODO: check 1 -> 5?

        if state in (State.EXTRACTION_SOURCE, State.MUSIC_DISCUSSION):
            if turn > 9:
                flag = 1

        else:
            print("over the turn")
            flag = 1

    return response, flag, slot
