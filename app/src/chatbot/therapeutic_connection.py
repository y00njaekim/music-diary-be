from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt, eval_prompt
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser


# for state unit test
from dotenv import load_dotenv

load_dotenv()


def print_memory_summary(memory):
    print("\n===== Memory summary =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[No summary stored yet]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """Information that must be extracted from the user's responses"""

    name: Optional[str] = Field(default=None, description="user name")

    difficulty: Optional[str] = Field(default=None, description="The difficulty the user is currently experiencing in their daily life")

    difficulty_category: Optional[str] = Field(
        default=None,
        description="A classification of the difficulty the user is currently facing. "
        "1) negative_experience: past bad memories or internal personal issues "
        "2) depression: issues related to the user's emotions "
        "3) communication: issues related to others or social groups",
        enum=["negative_experience", "depression", "communication"],
    )

    motivation: Optional[str] = Field(default=None, description="What the user hopes to gain through music therapy")


def therapeutic_connection(user_input, llm, memory, pre_slot) -> str:

    # 1. Prompt setup with few-shot examples
    therapeutic_connection_question = """
    [Self-introduction Task]
    - Exchange introductions with the user.
    Examples:
    - Hello, I am Soul, your partner in creating music. Shall we begin our songwriting journey together?

    [Understanding Goals and Motivation Task]
    - Ask whether music has comforted the user in the past.
    - Learn what kinds of music have helped them.
    - Discover what they hope to gain and their motivation for joining music therapy.
    Examples:
    - Are you feeling anxious about the music activity today?
    - Are you looking forward to the session?
    - Are you ready to enjoy making music together?

    [Identifying Difficulties Task]
    - Understand the challenges the user is currently facing.
    - Discuss the situations that trigger those difficulties and the problems that follow.
    - Encourage the user to express their emotions.
    Examples:
    - What has been the hardest part of your daily life lately?
    - Could you describe in more detail when those difficult moments arise?
    """
    full_few_shot_dialogue = """
    Follow the tone and pacing of the dialogue below.
    <Sample Dialogue>
    AI: Hello, I am Soul, here to create music with you. Shall we get started?
    Human: Hello. Yes, thank you for guiding me.
    AI: Were you looking forward to today's session? Are you feeling nervous or worried about anything?
    Human: I am a bit nervous. I am not sure I will be able to make good music.
    AI: That feeling is completely natural. Today, let's focus on expressing your heart rather than perfection. Have you ever felt comforted by music before?
    Human: Yes, calm piano music helps me settle down whenever things are tough.
    AI: That music sounds like it has been a source of comfort for you. What would you like to gain from making music today?
    Human: I want to express my feelings more honestly because I usually keep them inside.
    AI: That intention is meaningful. What has been the hardest part of your daily life lately?
    Human: I stay busy and overwhelmed, yet I feel no sense of accomplishment. I also feel distant from people.
    AI: That must be exhausting. When those situations repeat, it is natural for your heart to feel heavy. In what moments do you feel the distance or emptiness most strongly?
    Human: After talking with people and then being alone. I start wondering if I did something wrong and overthink everything.
    AI: That is such a lonely and difficult feeling. How about we place those emotions into the music today? Even if words are hard, the music can hold them for you.
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history","pre_slot"],
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
    question = question_chain.invoke({"user_message": user_input, "history": history,"pre_slot": pre_slot})

    # eval_chain = eval_prompt | llm | StrOutputParser()
    # eval = eval_chain.invoke({"history": history, "question": question, "bot_questions": bot_question, "user_input": user_input})
    memory.save_context({"input": user_input}, {"output": question})

    print_memory_summary(memory)
    # print("---")
    # print("Condition ", bot_question)
    # print("CoVe Q: ", question)
    # print("----")
    retrieved_memory_variables = memory.load_memory_variables({})
    current_chat_history = retrieved_memory_variables.get("history", "")  # Fetch the history value if it exists

    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": current_chat_history}))

    return question, slot
