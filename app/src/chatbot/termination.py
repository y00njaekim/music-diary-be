from llm_instance import llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def termination(total_dialogue: str, llm, pre_slot) -> str:
    termination_question="""
    [Termination Task]
    - 사용자에게 오늘 즐거웠다고 말하며 끝냅니다.
    - 질문이 아닌 잘가라는 인사를 진행합니다. 
    Example:
    - "오늘 정말 즐거운 시간이었습니다. 다음에 다시 만나요."
    - "오늘 나눠주신 말씀 덕분에 유익한 시간되었습니다."
    - "저는 항상 여기 있으니 언제든 찾아주세요."
    """

    question_prompt = PromptTemplate(template=termination_question)
    question_chain = question_prompt | llm | StrOutputParser()
    question = question_chain.invoke({})

    history_summary_prompt="""
    [Dialogue Summary Task]
    - 사용자와 챗봇의 대화기록을 보고 내용을 3~4줄로 요약하세요. 
    - 사용자가 나눈 어려움, 음악 컨셉, 음악정보등을 중점적으로 요약하면 됩니다.

    Example: 
    - “무기력함과 외로움”이라는 감정을 기반으로, “혼자 있는 나를 이해하고 위로하는” 주제의 가사와 음악을 창작했어요.
    - “타인의 시선에 대한 불안”을 바탕으로, “있는 그대로의 나를 받아들이는” 내용의 창작물을 만들었어요.
    - “반복되는 일상 속 공허함”이라는 어려움을 토대로, “작은 것에서 의미를 찾는” 주제로 가사와 음악을 완성했어요
    [유의사항]
    - 3~4줄로 요약하세요. (- < 포함하지 않고 줄글로)
    - 문해력이 안좋은 청각장애인이 읽을 글이니 최대한 쉬운 단어를 사용하세요.
    - 이 대화는 음악치료를 위해 음악을 만드는 과정에서 나온 대화입니다. 세션마다 반복되는 말 같은 것들은 생략하세요. 

    """
    summary_prompt = PromptTemplate(
        input_variables=["total_dialogue","pre_slot"],
        template=history_summary_prompt
        + "\n"
        + "아래를 보고 참고하여 응답을 생성하세요."
        + "Previous Slot: {pre_slot}\n"
        + "Chat history: {total_dialogue}\n"
    )

    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"total_dialogue": total_dialogue,"pre_slot": pre_slot })
    
    
    return summary+'\n'+question, summary