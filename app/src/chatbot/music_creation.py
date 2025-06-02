from langchain.prompts import PromptTemplate
from .prefix import question_prefix_prompt, slot_prefix_prompt
from pydantic import BaseModel, Field
from typing import Optional


from langchain_core.output_parsers import StrOutputParser


def print_memory_summary(memory):
    print("\n===== 💬 요약된 memory 내용 =====")
    memory_vars = memory.load_memory_variables({})
    summary = memory_vars.get("history", "[현재 저장된 요약 없음]")
    print(summary)
    print("================================\n")


class OutputFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""

    music_information: Optional[str] = Field(default=None, description="Detailed information about the user's usual music activities")

    genre: Optional[str] = Field(default=None, description="The genre of music the user wants to create")

    instrument: Optional[str] = Field(default=None, description="Instruments to be included in the music")

    mood: Optional[str] = Field(default=None, description="The mood or atmosphere of the music")

    vocal: Optional[str] = Field(default=None, description="Information about the desired vocalist or vocal style")

    tempo: Optional[str] = Field(default=None, description="The tempo or speed of the music")


def music_making(user_input, llm, memory):
    extraction_source_question = f"""
    가사는 수정하지 않습니다. 
    [Music Preference Task]
    - 사용자가 최근에 한 음악 활동에 대해 질문합니다.
    - 평소 사용자가 경험하는 음악활동에서 사용하는 음악과 비슷한 음악의 정보를 수집해 사용자가 선호하는 음악을 알기 위함입니다.
    Example:
    - 최근 어떤 음악 활동(음악감상, 악기연주, 노래부르기 등)을 했나요?
    - 최근 어떤 음악을 주로 감상했나요?
    - 평소 좋아하는 음악이 무엇인가요?
    - 그 음악에 대해서 저에게 설명해줄 수 있나요?
    - 혹시 싫어하거나 제외했으면 하는 음악이 있나요?

    [Making Music Task]
    - 만들고 싶은 음악의 세부사항을 정합니다.
    - 사용자가 어려워하는 경우에는 추천하여 진행합니다.
    Example:
    - 위의 가사를 가지고 어떤 장르/스타일의 음악으로 만들면 좋을까요?
    - 노래의 빠르기는 어떻게 할까요? 
    - 반주 악기는 어떤 것을 선호하나요?
    - 음악의 전반적인 분위기는 어떻게 할까요? 
    - 가수의 음색은 어땠으면 좋을까요?
    - 가수의 성별은 어땠으면 좋을까요?
    - 이 주제에 맞는 음악요소를 추천 받고 싶나요? 
    """
    full_few_shot_dialogue = """
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 최근 어떤 음악 활동을 하셨나요? 예를 들면 음악 감상, 노래 부르기, 악기 연주 등이 있을 수 있어요.
    Human: 요즘 스트레스 해소하려고 음악 감상을 자주 해요.
    AI: 그렇군요! 요즘 주로 어떤 음악을 감상하시나요?
    Human: 잔잔한 피아노 연주곡이나 인디 감성의 노래들을 자주 들어요.
    AI: 평소 좋아하는 음악 장르는 어떤 스타일인가요?
    Human: 감성적인 발라드나 어쿠스틱 느낌의 음악이 좋아요.
    AI: 혹시 반대로 싫어하거나 피하고 싶은 음악 장르는 있을까요?
    Human: 네, 너무 시끄러운 락이나 EDM은 잘 안 듣게 되더라고요.
    AI: 감사합니다! 말씀해주신 취향을 바탕으로 음악을 함께 만들어볼게요.
    AI: 앞에서 이야기한 감성적인 분위기를 살려서 노래를 만든다면, 어떤 장르가 어울릴까요? 어쿠스틱 발라드, 포크 스타일 등 추천해드릴까요?
    Human: 어쿠스틱 발라드 좋을 것 같아요.
    AI: 노래의 빠르기는 느린 템포가 괜찮을까요, 아니면 약간 리듬감 있게 중간 정도로 가볼까요?
    Human: 너무 느리진 않고 중간 템포였으면 해요.
    AI: 반주에 어떤 악기를 중심으로 하면 좋을까요? 피아노, 기타, 현악기 등이 있어요.
    Human: 어쿠스틱 기타랑 피아노 조합이 좋을 것 같아요.
    AI: 음악의 분위기는 따뜻하고 위로가 되는 느낌일까요, 아니면 더 쓸쓸하고 내면적인 느낌일까요?
    Human: 따뜻하고 위로가 되는 분위기로 하고 싶어요.
    AI: 좋습니다! 마지막으로, 노래를 부를 가수의 음색과 성별은 어떻게 상상하고 계세요?
    Human: 감성적이고 맑은 음색이면 좋겠어요. 여성보컬이면 더 좋아요. 저는 여성보컬은 조금 더 잘 들리거든요. 
    """
    question_prompt = PromptTemplate(
        input_variables=["user_message", "history"],
        template=question_prefix_prompt
        + "\n"
        + extraction_source_question
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

    memory.save_context({"input": user_input}, {"output": question})

    # print_memory_summary(memory)

    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": history}))

    return question, slot


def music_creation(user_input):

    return "[music_creation 단계입니다.]"
