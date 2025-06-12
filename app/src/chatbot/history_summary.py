from llm_instance import llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def history_summary(total_dialogue: str, llm) -> str:

    history_summary_prompt="""
    [Dialogue Summary Task]
    사용자와 챗봇의 대화기록을 보고 내용을 3~4줄로 요약하세요. 
    사용자가 나눈 어려움, 음악 컨셉, 음악정보등을 중점적으로 요약하면 됩니다.

    대화기록: {total_dialogue}

    [유의사항]
    - 3~4줄로 요약하세요.
    - 문해력이 안좋은 청각장애인이 읽을 글이니 최대한 쉬운 단어를 사용하세요.
    - 이 대화는 음악치료를 위해 음악을 만드는 과정에서 나온 대화입니다. 세션마다 반복되는 말 같은 것들은 생략하세요. 
    """
    summary_prompt = PromptTemplate(
        input_variables=["total_dialogue"],
        template=history_summary_prompt
        + "\n"
        + "Chat history: {total_dialogue}\n"
    )

    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"total_dialogue": total_dialogue})
    
    
    return summary