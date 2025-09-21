from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt, eval_prompt
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

class OutputFormat(BaseModel):
    """Information that must be extracted from the user's responses"""
    name: Optional[str] = Field(default=None, description="user name")
    experience: Optional[str] = Field(default=None, description="The user's experience from today")
    difficulty: Optional[str] = Field(default=None, description="The difficulty the user is currently experiencing in their daily life")
    expression:  Optional[str] = Field(default=None, description="A detailed expression of the user's emotions, experiences, and thoughts")

def therapeutic_connection_re(user_input, summary, llm, memory, pre_slot) -> str:
    therapeutic_connection_question="""
    [Session Review]
    Use the summary below to briefly review the creation from the previous session.
    Summary:
    {summary}
    Example:
    - Based on the emotions of "lethargy and loneliness," we wrote lyrics and music about comforting the self who is alone.
    - Centered on the anxiety of "being watched by others," we made a piece about accepting myself as I am.
    - From the struggle of "emptiness in a repetitive routine," we completed lyrics and music about finding meaning in small things.

    [Rapport Building Task]
    Talk casually about the user's day so they can organize the emotions, events, and thoughts they experienced.
    Ask for more detail when they mention emotions.
    Respond with empathy when they share events or thoughts.
    Example:
    - Hello! How was your day?
    - (If they describe an emotion) Could you tell me more about what made you feel that way?

    [Identifying Difficulties Task]
    After exploring the feelings, events, or thoughts that mattered most today, clarify the challenges the user is currently facing.
    Use that insight to discover a musical theme and motivate them to continue with songwriting.
    Example:
    - You mentioned ~ about your day. Shall we talk more about that theme?
    - Is there any topic you would especially like to explore together?
    - How about we create music based on this story?
    """
    full_few_shot_dialogue="""
    AI: In our last session we talked about "unspoken emotions" and wrote lyrics and music about releasing the words locked in your heart.
    AI: How was your day today?
    Human: It was just tiring. Nothing special happened, but I felt drained.
    AI: Sounds like an exhausting day. Could you share a bit more about what led to that feeling?
    Human: There was no specific reason. I just lacked motivation from the morning, and even talking to people felt bothersome.
    AI: We all have days like that. Sometimes emotions feel heavy without a clear event. What thoughts were taking up most of your mind today?
    Human: I kept wondering, "Is it okay to keep living like this?" I constantly wanted to escape the repetitive routine.
    AI: I can relate to that. When we do not have room to reflect on ourselves, those questions naturally appear. The fact that you felt that way today means it is already an important story.
    AI: What you said earlier, "Is it okay to keep living like this," really stood out. Would you like to unpack that topic a little more with me?
    Human: Sure, it is something I think about often. I am doing my best, but it still feels like I am missing something.
    AI: That feeling shows just how much you are reflecting. Is there any other emotion or situation you would like to talk about?
    Human: Well... when I am alone I suddenly feel lonely and wonder if I am doing something wrong.
    AI: Those feelings are deep and important. How about we turn the emotions and thoughts we are sharing right now into music together?
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history", "summary","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + therapeutic_connection_question
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
    question = question_chain.invoke({"user_message": user_input, "history": history, "summary":summary, "pre_slot": pre_slot})

    memory.save_context({"input": user_input}, {"output": question})

    retrieved_memory_variables = memory.load_memory_variables({})
    current_chat_history = retrieved_memory_variables.get("history", "")  # Fetch the history value if it exists
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": current_chat_history}))

    return question, slot
