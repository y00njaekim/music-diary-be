from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt, eval_prompt
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser


# for state unit test
from dotenv import load_dotenv

load_dotenv()


def print_memory_summary(memory):
    print("\n===== 💬 요약된 memory 내용 =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[현재 저장된 요약 없음]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""

    name: Optional[str] = Field(default=None, description="user name")

    difficulty: Optional[str] = Field(default=None, description="The difficulty the user is currently experiencing in their daily life")

    difficulty_category: Optional[str] = Field(
        default=None,
        description="A classification of the difficulty the user is currently facing. "
        "1) negative_experience: past bad memories or internal personal issues "
        "2) depression: issues related to the user’s emotions "
        "3) communication: issues related to others or social groups",
        enum=["negative_experience", "depression", "communication"],
    )

    motivation: Optional[str] = Field(default=None, description="What the user hopes to gain through music therapy")


def therapeutic_connection(user_input, llm, memory) -> str:

    # 1. 질문 + few-shot
    therapeutic_connection_question = """
    [Self-introduction Task]  
    - Exchange introductions with the user.  
    Example:  
    - 안녕하세요, 저는 당신과 함께 음악을 만들어갈 쏘울(Soul)입니다. 함께 음악만들기 활동을 시작해볼까요?

    [Understanding Goals and Motivation Task]  
    - 사용자가 음악을 통해 위안을 받은 적이 있는지 물어보세요. 
    - 어떤 종류의 음악이 도움이 되었는지 물어보세요.
    - 음악 치료를 통해 얻고 싶은 것과 참여 동기를 알아보세요. 
    Examples:  
    - 오늘 음악 활동에 대해 걱정되시나요? 
    - 오늘 음악 세션을 기대하시나요? 
    - 음악을 만들며 즐거운 시간을 보내실 준비가 되셨나요?

    [Identifying Difficulties Task]  
    - 사용자가 현재 겪고 있는 어려움을 파악합니다.
    - 어떤 상황이 그 어려움을 유발하는지, 그로 인해 어떤 문제가 생기는지를 좀 더 자세히 이야기합니다.
    - 사용자가 자신의 감정을 표현할 수 있도록 도와줍니다.
    Examples:  
    - 요즘 일상에서 가장 힘들다고 느끼는 점은 무엇인가요?
    - 어떤 상황에서 힘들어지고 있는지 조금 더 자세히 말해줄 수 있나요?
    """
    full_few_shot_dialogue = """
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 안녕하세요, 저는 당신과 함께 음악을 만들어갈 쏘울(Soul)입니다. 함께 음악 만들기 활동을 시작해볼까요?
    Human: 안녕하세요. 네, 잘 부탁드려요.
    AI: 오늘 음악 세션을 기대하고 계셨나요? 혹시 걱정되거나 긴장되는 부분도 있으실까요?
    Human: 조금 긴장돼요. 제가 음악을 잘 만들 수 있을지 걱정도 되고요.
    AI: 그렇게 느끼는 건 아주 자연스러워요. 오늘은 완성도보다는 마음을 표현하는 데 집중하면 좋을 것 같아요. 혹시 이전에 음악을 통해 위로를 받은 경험이 있으신가요?
    Human: 네, 힘들 때마다 잔잔한 피아노 음악을 들으면 마음이 조금 가라앉더라고요.
    AI: 그런 음악이 사용자님께 위안이 되었군요. 오늘 음악 만들기 활동을 통해 어떤 것을 얻고 싶으세요?
    Human: 제 감정을 좀 더 솔직하게 표현해보고 싶어요. 평소에는 잘 안 드러내니까요.
    AI: 그 마음, 정말 소중하게 느껴져요. 그럼 요즘 일상에서는 어떤 점이 가장 힘들게 느껴지세요?
    Human: 그냥 계속 바쁘고 정신없는데도 성취감이 없어요. 사람들하고도 자꾸 거리감이 느껴지고요.
    AI: 많이 지치셨겠어요. 그런 상황이 반복되면 마음이 무거워질 수밖에 없죠. 어떤 순간에 특히 그런 거리감이나 공허함이 더 크게 느껴지시나요?
    Human: 사람들이랑 얘기하고 나서 혼자 있을 때요. 괜히 내가 잘못했나 싶고 생각이 많아져요.
    AI: 그런 생각이 드는 건 참 외롭고 힘든 일이죠. 오늘 음악 속에 그 감정들을 조금씩 담아보면 어떨까요? 말로는 어렵지만 음악으로는 충분히 표현할 수 있을 거예요.
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history"],
        template=question_prefix_prompt
        + "\n"
        + therapeutic_connection_question
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "Chat history: {history}\n"
        + "User said: {user_message}",
    )

    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")

    question_chain = question_prompt | llm | StrOutputParser()
    question = question_chain.invoke({"user_message": user_input, "history": history})

    # eval_chain= eval_prompt | llm | StrOutputParser()
    # eval=eval_chain.invoke({"history": history, "question": question, "bot_questions": bot_question,"user_input":user_input,})

    memory.save_context({"input": user_input}, {"output": question})

    # print_memory_summary(memory)
    # print("---")
    # print("조건 ", bot_question)
    # print("CoVe Q: ", question)
    # print("----")
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return question, slot
