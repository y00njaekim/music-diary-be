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

# Load environment variables from .env
load_dotenv()

# Configure the Mureka API endpoint and key
mureka_api_endpoint = "https://api.mureka.ai"
mureka_api_key = os.getenv("MUREKA_API_KEY")

# TODO: add an instrumental version as well.
def print_memory_summary(memory):
    print("\n===== Memory summary =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[No summary stored yet]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """Information that must be extracted from the user's responses"""

    music_information: Optional[str] = Field(default=None, description="Detailed information about the user's usual music activities")

    genre: Optional[str] = Field(default=None, description="The genre of music the user wants to create")

    instrument: Optional[str] = Field(default=None, description="Instruments to be included in the music")

    mood: Optional[str] = Field(default=None, description="The mood or atmosphere of the music")

    vocal: Optional[str] = Field(default=None, description="Information about the desired vocalist or vocal style")

    tempo: Optional[str] = Field(default=None, description="The tempo or speed of the music")

    title: Optional[str] = Field(default=None, description="The title of the music that you specified")


def music_making(user_input, llm, memory, pre_slot):
    extraction_source_question = f"""
    Keep the lyrics exactly as they are.
    [Making Music Task]
    - Decide on the musical details the user wants.
    - Offer suggestions when the user feels unsure.
    Example:
    - Which genre or style would fit these lyrics?
    - How fast should the song be?
    - Which instruments would you like in the accompaniment?
    - What overall mood should the music have?
    - What kind of vocal tone do you imagine?
    - Do you have a preference for the singer's gender?
    - Would you like suggestions for musical elements that suit this theme?
    - What title would you like for the song?
    """
    full_few_shot_dialogue = """
    Follow the tone and pacing of the dialogue below.
    <Sample Dialogue>
    AI: To keep the sentimental mood we discussed, which genre do you think fits? Would you like suggestions such as acoustic ballad or folk?
    Human: An acoustic ballad sounds good.
    AI: Should the tempo stay slow, or do you prefer a medium pace with a bit of groove?
    Human: I would like a medium tempo, not too slow.
    AI: Which instruments should lead the accompaniment? Piano, guitar, and strings are options. Piano delivers delicate emotion and suits a calm mood. Guitar adds warmth and fits acoustic or indie vibes. Strings bring depth and richness to the feelings. Which instrument best matches how you want it to sound?
    Human: A mix of acoustic guitar and piano would be great.
    AI: Should the overall mood feel warm and comforting, or more introspective and somber?
    Human: I want it to feel warm and comforting.
    AI: Perfect! Lastly, what kind of vocal tone and gender do you imagine for the singer?
    Human: I would like a clear and emotional tone. A female vocalist works best for me.
    AI: Great. What would you like to call the song?
    Human: Something that sounds like words of comfort.
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + extraction_source_question
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "Generate the next question using the references below."
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
    """Look up the status of the specified task ID via the Mureka API."""
    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
    }
    response = requests.get(mureka_api_endpoint + f"/v1/song/query/{id}", headers=headers)
    response.raise_for_status()
    return response.json()


def generate_mureka_song_and_wait(title: str, lyrics: str, music_component: str) -> str:
    """Request song generation from the Mureka API, wait for completion, and return the audio URL."""
    print(f"Title: {title}")
    print(f"Music style: {music_component}")

    # 1. Submit the generation request (POST)
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
        print(f"Song generation started. Task ID: {task_id}")

        # 2. Poll until the task completes
        retry_delay = 5  # seconds between checks
        max_retries = 100  # roughly eight minutes
        retry_count = 0

        while retry_count < max_retries:
            task_status_response = query_mureka_task(task_id)
            status = task_status_response.get("status")

            if status == "succeeded":
                audio_url = task_status_response["choices"][0]["url"]
                print(f"Song generation succeeded. Audio URL: {audio_url}")
                return audio_url
            elif status == "failed":
                print(f"Task failed: {task_status_response}")
                return "Task failed"
            else:
                # Status is "processing" or something similar
                print(f"Task in progress (status: {status}). Retrying in {retry_delay} seconds.")
                time.sleep(retry_delay)
                retry_count += 1

        print("Maximum retry count exceeded. Task timed out.")
        return "Task timed out"

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return f"API Error: {e}"


def music_creation(user_input, llm, memory, pre_slot):
    """Extract lyrics and musical style from the CombinedSlot input, generate a song with Mureka, and return the audio URL."""
    # Parse string inputs into dictionaries
    if isinstance(user_input, str):
        user_input_dict = json.loads(user_input)
    # 1. Extract lyrics
    lyrics = user_input_dict.get("lyrics", None)
    if not lyrics:
        response = "Lyrics were not provided."
        # Save history and regenerate slot
        memory_vars = memory.load_memory_variables({})
        history = memory_vars.get("history", "")
        structured_llm = llm.with_structured_output(schema=OutputFormat)
        slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
        slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))
        return response, slot

    # 2. Build the musical style prompt
    style_elements = []
    for key in ["genre", "instrument", "mood", "vocal", "tempo"]:
        value = user_input_dict.get(key, None)
        if value:
            # TODO: Mureka metadata probably expects tag1, tag2, ... instead of key:value.
            # style_elements.append(f"{key}: {value}")
            style_elements.append(value)
    music_component = ", ".join(style_elements) if style_elements else ""

    # 3. Extract the title (defaults to "Untitled Song")
    # TODO: ensure the slot uses "title" rather than the user's name.
    title = user_input_dict.get("title", "Untitled Song")

    # 4. Call the Mureka API and return the result
    audio_url = generate_mureka_song_and_wait(title, lyrics, music_component)

    if audio_url.startswith("http"):
        response = f"The song was generated successfully.\nAudio file: {audio_url}"
    else:
        response = f"Song generation failed: {audio_url}"

    # Save the interaction to memory
    memory.save_context({"input": user_input}, {"output": response})
    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")

    # Create the slot with the structured LLM
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return response, slot
