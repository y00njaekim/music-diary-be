from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt, eval_prompt
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

class OutputFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""
    name: Optional[str] = Field(default=None, description="user name")
    experience: Optional[str] = Field(default=None, description="The user's experience from today")
    difficulty: Optional[str] = Field(default=None, description="The difficulty the user is currently experiencing in their daily life")
    expression:  Optional[str] = Field(default=None, description="A detailed expression of the user's emotions, experiences, and thoughts")

def therapeutic_connection_re(user_input, summary, llm, memory, pre_slot) -> str:
    therapeutic_connection_question="""
    [Session Review]
    {summary}를 보고 전 회기의 창작물을 간단하게 리뷰합니다.
    Example: 
    - “무기력함과 외로움”이라는 감정을 기반으로, “혼자 있는 나를 이해하고 위로하는” 주제의 가사와 음악을 창작했어요.
    - “타인의 시선에 대한 불안”을 바탕으로, “있는 그대로의 나를 받아들이는” 내용의 창작물을 만들었어요.
    - “반복되는 일상 속 공허함”이라는 어려움을 토대로, “작은 것에서 의미를 찾는” 주제로 가사와 음악을 완성했어요

    [Rapport Building Task]
    사용자가 스스로 느낀 감정과 사건, 생각을 정리할 수 있도록 오늘 하루에 대해서 가볍게 이야기하세요.
    사용자가 감정을 이야기할 경우 더 자세히 물어봅니다.
    사용자가 사건, 생각에 대해 이야기할 경우 공감적으로 진행합니다. 
    Example:  
    - 안녕하세요! 오늘 하루는 어땠나요?
    - (감정에 대해서 이야기) 왜 이런 감정이 들었는지 구체적으로 이야기해줄 수 있나요?

    [Identifying Difficulties Task]
    앞서 하루에 대해서 중요하다고 생각하는 감정/사건/생각에 대해서 좀 더 이야기한 후, 
    사용자가 현재 겪고 있는 어려움을 파악합니다. 이를 기반으로 음악적 주제를 발견하고 음악 창작으로 나아갈 수 있게 동기를 부여하세요. 
    Example:  
    - 오늘 하루 ~~라고 했어요. 이 주제에 대해서 더 이야기해볼까요?
    - 특히, 저와 더 이야기하고 싶은 주제가 더 있을까요?
    - 그럼 이 이야기를 바탕으로 음악을 만들어 볼까요?
    """
    full_few_shot_dialogue="""
    AI: 지난 세션에서는 “표현하지 못한 감정”에 대한 이야기를 통해, “마음속에 쌓인 말을 풀어내는” 가사와 음악을 함께 작업했어요.
    AI: 오늘 하루는 어땠나요?
    Human: 그냥 좀 지치는 하루였어요. 별일은 없었는데 기운이 없더라고요.
    AI: 지친 하루였군요. 왜 그런 감정이 들었는지 조금 더 이야기해주실 수 있을까요?
    Human: 특별한 이유는 없는데, 아침부터 의욕이 없었고 사람들하고 얘기하는 것도 귀찮게 느껴졌어요.
    AI: 그런 날이 있어요. 특별한 사건이 없어도 감정이 무겁게 느껴질 때가 있죠. 혹시 오늘 어떤 생각들이 머릿속을 가장 많이 차지했나요?
    Human: ‘이렇게 계속 지내도 괜찮을까?’ 하는 생각이요. 반복되는 일상에서 벗어나고 싶다는 생각이 자꾸 들었어요.
    AI: 그 생각, 정말 공감돼요. 반복되는 일상 속에서 나 자신을 돌아볼 여유가 없으면 그런 물음이 생기곤 하죠. 오늘 하루 그런 감정이 드셨다면, 그것만으로도 충분히 중요한 이야기예요.
    AI: 조금 전 “이렇게 계속 지내도 괜찮을까”라는 말씀이 인상 깊었어요. 이 주제에 대해서 저와 조금 더 이야기해볼까요?
    Human: 네, 그런 생각을 자주 하게 돼요. 그냥 열심히 살고는 있는데, 뭔가 놓치고 있는 느낌도 들고요.
    AI: 그 감정은 많은 걸 고민하고 있다는 증거 같아요. 혹시 저와 더 이야기하고 싶은 감정이나 상황이 더 있을까요?
    Human: 음… 혼자 있을 때 괜히 외로워지고, 내가 뭔가 잘 못하고 있는 건 아닌가 싶은 생각이 들 때가 많아요.
    AI: 그런 감정은 참 깊고 중요한 이야기예요. 우리가 지금 나누고 있는 이 대화 속 감정들, 생각들… 이걸 바탕으로 음악을 한번 만들어볼까요?
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history", "summary","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + therapeutic_connection_question
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "아래를 보고 참고하여 질문을 생성하세요."
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
    current_chat_history = retrieved_memory_variables.get("history", "") # 'history' 키로 값을 가져오고, 없으면 빈 문자열
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": current_chat_history}))

    return question, slot