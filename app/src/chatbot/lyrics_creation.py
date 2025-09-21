from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt
from pydantic import BaseModel, Field
from typing import Optional


from langchain_core.output_parsers import StrOutputParser


def print_memory_summary(memory):
    print("\n===== Memory summary =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[No summary stored yet]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """Information that must be extracted from the user's responses"""

    concept: Optional[str] = Field(default=None, description="The story or theme the user wants to express in the lyrics")

    concept_discussion: Optional[str] = Field(
        default=None, description="A summary of what the user shared about their concept during the [Concept Discussion Task]"
    )

    lyric_keyword: Optional[str] = Field(default=None, description="The main keyword that comes to mind when expressing the intended theme")

    lyric_image: Optional[str] = Field(default=None, description="The image that comes to mind when thinking of the lyrics")

    lyrics_content: Optional[str] = Field(default=None, description="Detailed sentences the user wants to include in the lyrics")


def extraction_source(user_input, llm, memory, pre_slot):
    extraction_source_question = f"""
    [Making Concept Task]
    - Decide on the theme for the song lyrics.
    Examples:
    - What story would you like the song to tell?
    - Which emotions or situations would you like to capture through the music?

    [Concept Discussion Task]
    - Explore the concept in greater detail.
    - Ask why the user wants to share this story and how it makes them feel when they talk about it.
    - Invite the user to look closely at their emotions.
    Examples:
    - Why do you want to talk about this theme?
    - Is there someone you would like to share these lyrics with?
    - How do you feel when you think about this story?
    - When you think about this story, what would you like to tell yourself?

    [Making Lyrics Task]
    - Gather information needed to write the lyrics.
    - Ask for the core ideas the user wants to include.
    - Encourage them to share as many ideas as possible.
    - Visual metaphors are a helpful approach.
    - When making suggestions, offer options the user can choose from and explain why they fit the current lyrics.
    Examples:
    - Which words come to mind when you express this theme?
    - What feelings arise when you think about it?
    - Short phrases or single words are fine. Share anything that comes to mind.
    - What imagery do you see when you imagine this theme?
    - What colors do you associate with it?
    - Is there a place that fits this theme?
    """
    full_few_shot_dialogue = """
    Follow the tone and pacing of the dialogue below.
    <Sample Dialogue>
    AI: Let's create lyrics together. What story would you like the song to hold?
    Human: I want to talk about a bright future ahead of me.
    AI: What makes you want to tell that story?
    Human: I want to reassure myself that I can do well moving forward and leave the painful memories behind.
    AI: That is so important. When you think of those past memories, what emotions surface?
    Human: I wonder why I acted that way back then.
    AI: That is a natural reaction. I feel something similar when I look back, too. But focusing on the future instead of being swallowed by the past is a wonderful direction. Do any keywords come to mind when you recall this story?
    Human: Phrases like "leave the bad memories behind" and "start again" come to mind.
    AI: Those are powerful keywords. Would you like to draft some lyrics with them? Do any lines appear when you think about it?
    Human: Hmm... that feels difficult.
    AI: Then let's imagine some images that fit those keywords, like a rising sun or the first light of dawn. They capture the feeling of moving past the darkness toward a new beginning, just like you described. Do any of those images resonate with you, or is there another image you prefer?
    Human: The rising sun feels right.
    AI: The rising sun suits the idea of a bright future. When you picture it, do any short lines or phrases come to mind for the lyrics?
    Human: Maybe something like, "Leave the pain behind, trust myself from now on, shine again, tomorrow will be different."
    AI: Great! I will craft the lyrics using this sentence, your keywords, and the imagery you shared.
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

    memory.save_context({"input": user_input}, {"output": question})

    # print_memory_summary(memory)

    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return question, slot


class LyricsOutput(BaseModel):
    """Information that must be extracted from the user's responses"""

    lyrics: Optional[str] = Field(default=None, description="AI-generated lyrics")


def making_lyrics(user_input, llm, memory, pre_slot):
    making_lyrics_prompt = f"""
    [Lyrics Generative Task]
    - Generate lyrics based on the collected slots.
    - Present the output using the format below.
    Output Format:
    [Verse]
    Sitting alone in the deep night
    Resting beneath the cold moonlight
    Searching for a breath in the silent breeze
    Wandering through hazy memories

    [Verse 2]
    Even the starlight turns away from me
    Longing wraps itself around my heart
    Only my footsteps echo on the empty street
    Carrying me along an endless road

    [Chorus]
    In the deep night, loneliness
    Makes my heart cry once more
    This heart soaked through with tears
    Holds a love no one else will ever know

    [Bridge]
    I whisper softly to the moon
    Who could ever know this ache?
    Will it disappear if I close my eyes?
    This song that never finds an end

    [Verse 3]
    Even when dawn arrives I cannot sleep
    I keep searching for you inside my dreams
    A voice carried to me by the wind
    Shakes me awake all over again

    [Chorus]
    In the deep night, loneliness
    Makes my heart cry once more
    This heart soaked through with tears
    Holds a love no one else will ever know
    """

    question_prompt = PromptTemplate(input_variables=["slot", "history"], template=making_lyrics_prompt + "\n" + "Chat history: {history}\n" + "slot: {slot}")

    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")

    question_chain = question_prompt | llm | StrOutputParser()
    question = question_chain.invoke({"slot": user_input, "history": history})

    # TODO: Confirm whether memory should be updated here to avoid duplicating save_context calls inside the function and in main.
    memory.save_context({"input": user_input}, {"output": question})
    slot = LyricsOutput(lyrics=question)

    return question, slot
