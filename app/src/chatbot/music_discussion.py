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

    individual_emotion: Optional[str] = Field(default=None, description="Emotion felt while experiencing the song created")

    change_mind: Optional[str] = Field(default=None, description="Aspects of thinking that changed through the music-making process")

    change_attitude: Optional[str] = Field(default=None, description="Changes in attitude toward difficulties through the music-making process")

    touching_lyrics: Optional[str] = Field(default=None, description="Lyrics that were particularly touching or resonant")

    strength: Optional[str] = Field(default=None, description="Personal strengths discovered through the music-making activity")

    feeling: Optional[str] = Field(default=None, description="Feelings after experiencing the music-making activity")


def music_discussion(user_input, llm, memory, pre_slot):
    music_discussion_question = f"""
    [Music Opinion Task]
    - Explore the user's overall satisfaction with the song they composed and identify the parts they enjoyed.
    Examples:
    - Are you satisfied with the music you created?
    - How did you want this song to portray who you are today?
    - Do you feel the song conveys your emotions, thoughts, and self-image?

    [Identity Task]
    - Discuss the lyrics and music to nurture a positive understanding of the user's identity. Find specific moments in the conversation that triggered emotional or cognitive change.
    - Reflect on what has shifted for them through the music-making activity.
    - Draw out the thoughts and feelings that changed as a result of finishing the song.
    - Help the user organize their inner thoughts by asking follow-up questions that invite them to open up.
    - When the user mentions memories or events, ask why those moments resurfaced and explore them more deeply.
    Examples:
    - How did creating this music help you work through the thoughts or feelings you were holding?
    - It sounds like this song carries something very important to you. Which part is that?
    - Why do you think you felt that way?
    - Did any past memories come to mind?
    - In what ways did your thoughts or feelings change through this process?
    - Do you think the song captured emotions or ideas you had not noticed before? If so, which parts?
    - Has your perspective or attitude toward your current challenges shifted?
    - How did the music-making experience influence the way you face those challenges?

    [Completion Task]
    - Close out the music-making session.
    - Ask the user how it felt to create the song together.
    Examples:
    - "How did the song you made today comfort you?"
    - "What would you like to say to the 'you' who is listening to these emotions right now?"
    - "Was this a meaningful time together? I am grateful I could listen to such a beautiful song today."
    """

    full_few_shot_dialogue = """
    Follow the tone and pacing of the example dialogue below.
    <Sample Dialogue>
    AI: I was deeply moved listening to the song you made today. Are you happy with how it turned out?
    Human: Yes, it came out better than I expected.
    AI: Which part do you like the most?
    Human: The melody line. I wanted it to capture the sentimental style I have been listening to lately.
    AI: That feeling definitely came through. When you made this song, how did you want it to portray who you are today?
    Human: Maybe... a bit tired and quiet. Someone who is not saying much but holding a lot inside.
    AI: I could sense that emotion naturally in the music. Do you feel the song expresses your current thoughts and feelings?
    Human: Yes, more than I thought. While writing it, I realized what I had been feeling.
    AI: So the realization surfaced as you were making the music. In what way did the process help you work through those emotions or thoughts?
    Human: I was able to face feelings I usually brush aside. I could place what I cannot say into the melody.
    AI: That sounds like a precious experience. Did any special memories or moments come to mind while composing it?
    Human: I remembered crying alone with my headphones on in high school. I think I listened to music back then to sort through my emotions too.
    AI: Those feelings from that time are still living in your heart. Why do you think that scene resurfaced now?
    Human: I think it is because I am dealing with similar feelings lately. I just want to lean on music like I did then.
    AI: Feeling that way is completely natural. Expressing your emotions through music like this takes a lot of courage. I sense that something very important to you is inside this song. What part do you think that is?
    Human: Probably the message that I am okay. I wanted to say that my emotions are fine just as they are.
    AI: That message is so touching. If you could share one last sentence with the version of you who made this song today, what would you say?
    Human: "You are okay right now, you are doing well." That is what I want to say.
    AI: Such warm words. I am glad the song comforted you today. Thank you for letting me experience your meaningful music.
    """

    question_prompt = PromptTemplate(
        input_variables=["user_message", "history","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + music_discussion_question
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
    retrieved_memory_variables = memory.load_memory_variables({})
    current_chat_history = retrieved_memory_variables.get("history", "")  # Fetch the history value if it exists
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": current_chat_history}))

    return question, slot
