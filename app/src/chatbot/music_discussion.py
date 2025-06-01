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

    individual_emotion: Optional[str] = Field(default=None, description="Emotion felt while experiencing the song created")

    change_mind: Optional[str] = Field(default=None, description="Aspects of thinking that changed through the music-making process")

    change_attitude: Optional[str] = Field(default=None, description="Changes in attitude toward difficulties through the music-making process")

    touching_lyrics: Optional[str] = Field(default=None, description="Lyrics that were particularly touching or resonant")

    strength: Optional[str] = Field(default=None, description="Personal strengths discovered through the music-making activity")

    feeling: Optional[str] = Field(default=None, description="Feelings after experiencing the music-making activity")


def music_discussion(user_input, llm, memory, var_dict, bot_question):
    extraction_source_question = f"""
    [Music Opinion Task]
    - 만든 음악의 특정 부분에 대해서 이야기를 나누는 과정
    - 자세하고 세심하게 질문하는 것이 핵심입니다.
    - following question을 통해 사용자의 속마음을 털어놓도록 하세요. 
    Examples:  
    - 특별히 어떤 가사(단어, 구절)가 와닿으세요? 그 이유는 무엇인가요?
    - 어떤 부분에서 위로가 되었나요? (가사/멜로디/분위기) 
    - 이 작업을 통해 자신의 강점을 발견하셨다면 어떤 것이 있을까요?
    - 이 노래가 당신의 감정이나 생각을 잘 전달했다고 생각하시나요? 
    
    [Reflection Task]
    - 음악만들기 활동을 통해 이전과 바뀌어진 점에 대해서 이야기합니다.
    - 사용자의 달라진 점에 대해 격려하고 질문하세요.
    Example:
    - 음악만들기 과정을 통해서 갖고 있던 생각이나 감정을 다루는데 어떤 도움을 받으셨나요?
    - 음악만들기 과정을 통해 생각이나 느낌이 달라진 부분이 있다면 어떤 것일까요? 
    - 이 활동이 어떤 변화를 이끌어 낼 수 있다고 생각하나요?
    - 현재 가진 어려움을 대하는 것에 대해 생각이나 대하는 자세가 바뀌었다고 생각하시나요?
    [Complete Task]
    - 음악만들기 활동을 종료하는 단계입니다.
    - 사용자에게 함께 음악을 만들어본 소감과 언제든 다시 만나자고 이야기하며 끝냅니다.
    Example:
    - 의미 있는 시간 되셨나요? 저는 오늘 덕분에 멋진 음악을 감상할 수 있는 뜻깊은 시간이었어요. 오늘 활동이 어떠셨나요?
    - 오늘 새로운 장점을 알게 되어서 기뻤습니다. 저도 장점을 찾아보려고 노력해보려고 합니다. 오늘 활동 어떠셨나요?
    - 활동 중에서 좋았던 부분은 무엇인가요? 
    - 활동 중에서 어려웠던 부분은 무엇인가요?
    - 제가 어떤 것을 더 도와드리면 좋을까요? 
    """

    full_few_shot_dialogue = """
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 우리가 만든 노래에서 특별히 어떤 가사나 문장이 가장 와닿으셨나요?
    Human: “다시 빛나 내일은 분명 달라”라는 문장이요.
    AI: 그 문장이 와닿은 이유는 무엇인가요?
    Human: 요즘 계속 제자리에 머무는 느낌이 들어서, 그 문장이 희망처럼 느껴졌어요.
    AI: 그런 감정을 느낄 때 누군가 대신 말해주는 것 같은 문장이 큰 위로가 되죠. 혹시 멜로디나 분위기에서도 위로를 느낀 부분이 있었나요?
    Human: 조용히 시작해서 점점 밝아지는 멜로디가 마음을 토닥이는 느낌이었어요.
    AI: 그 따뜻한 변화를 느끼셨다니 정말 기뻐요. 혹시 이 작업을 하면서 스스로의 강점을 새롭게 발견하신 점이 있나요?
    Human: 감정을 말로 표현하는 게 어렵다고 생각했는데, 이렇게 가사로 정리해보니 할 수 있다는 생각이 들었어요.

    AI: 이번 음악 만들기 과정이 감정이나 생각을 다루는 데 어떤 도움이 되었나요?
    Human: 마음속에만 담아두었던 걸 밖으로 꺼낼 수 있었던 시간이었어요.
    AI: 내면을 표현하는 건 정말 용기 있는 일이에요. 그런 점에서 큰 걸 해내신 거예요. 혹시 활동을 하면서 생각이나 느낌이 바뀐 부분이 있었나요?
    Human: 네, 나도 따뜻한 말들을 할 수 있고, 나 자신을 위로할 수 있다는 걸 느꼈어요.
    AI: 너무 소중한 깨달음이에요. 앞으로도 음악이 그런 당신의 도구가 되기를 바라요. 이 활동이 본인에게 어떤 변화를 이끌어줄 수 있을 것 같나요?
    Human: 앞으로 힘든 일이 생겨도 이렇게 감정을 표현해볼 수 있을 것 같아요.
    AI: 그 변화의 시작점이 되었으면 좋겠어요. 정말 잘 해내셨어요.
    AI: 오늘 함께한 음악 만들기 활동, 의미 있는 시간이 되셨나요? 저는 덕분에 진심이 담긴 아름다운 음악을 들을 수 있어서 너무 뜻깊었어요.
    Human: 네, 생각보다 더 치유받는 시간이었어요.
    AI: 그렇게 느끼셨다니 정말 다행이에요. 활동 중에서 특히 좋았던 부분이 있다면 어떤 거였나요?
    Human: 제가 떠올린 이미지로 가사가 만들어질 때 정말 신기하고 뿌듯했어요.
    AI: 맞아요, 그 장면이 가사로 살아나는 순간은 정말 특별하죠. 혹시 활동 중에 어려웠던 부분도 있었을까요?
    Human: 처음에는 감정을 말로 꺼내는 게 좀 힘들었어요.
    AI: 그럼에도 끝까지 솔직하게 표현해주셔서 감사합니다. 다음에 또 함께한다면 제가 어떤 걸 더 도와드리면 좋을까요?
    Human: 질문을 잘 던져줘서 편했어요. 다음에도 그렇게 해주세요.
    AI: 꼭 그렇게 할게요. 오늘 정말 수고 많으셨습니다. 언제든지 음악이 필요할 때, 저와 함께 다시 만들어봐요!
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
