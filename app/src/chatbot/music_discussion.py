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


def music_discussion(user_input, llm, memory, pre_slot):
    music_discussion_question = f"""
    [Music Opinion Task]
    - 사용자가 만든 음악에 대해서 전반적인 만족감을 이야기합니다. 어떤 부분이 마음에 들었는지에 대해서 파악하세요. 
    Examples:  
    - 창작된 결과물에 대해서 만족스럽나요? 
    - 이 곡에서 ‘오늘의 나’를 어떤 모습으로 표현하고 싶었나요?
    - 이 노래가 당신의 모습, 감정, 생각을 잘 전달했다고 생각하시나요?
    
    [Identity Task]
    - 사용자가 만든 가사와 음악에 대해 이야기하면서 사용자의 자기 정체성에 대한 긍정적 인식을 유도하는 과정입니다. 사용자가 만든 음악에 대해서 전반적으로 이야기를 나누면서 사용자의 감정/사고적 변화를 일으킨 특정 부분을 발견하세요.     
    - 음악만들기 활동을 통해 이전과 바뀌어진 점에 대해서 이야기합니다.
    - 사용자가 음악 결과물을 통해 변화된 생각과 감정을 이끌어내는 것이 중요합니다. 
    - 사용자 스스로 생각과 감정을 정리할 수 있도록 following question을 통해 사용자의 속마음을 털어놓도록 하세요. 
    - 사용자가 특정 기억과 사건에 대해서 말한다면 왜 이러한 과거가 떠올랐는지 좀 더 깊게 이야기를 이끌어내주세요.
    Example:
    - 음악만들기 과정을 통해서 갖고 있던 생각이나 감정을 다루는데 어떤 도움을 받으셨나요?
    - 이 노래에 당신이 정말 중요하게 생각하는 무언가가 담긴것 같아요. 어떤 부분인가요? 
    - 왜 그렇게 느끼셨나요? 
    - 떠오른 과거의 기억이 있나요?
    - 음악만들기 과정을 통해 생각이나 느낌이 달라진 부분이 있다면 어떤 것일까요? 
    - 자신도 몰랐던 감정이나 생각이 노래에 담겼다고 느끼시나요? (있다면 어떤 부분인가요?)
    - 현재 가진 어려움을 대하는 것에 대해 생각이나 대하는 자세가 바뀌었다고 생각하시나요?
    - 현재 가진 어려움을 대하는 것에 대해 생각이나 대하는 자세가 바뀌었다고 생각하시나요?
    - 음악만들기 과정을 통해서 갖고 있던 생각이나 감정을 다루는데 어떤 도움을 받으셨나요?

    [Complete Task]
    - 음악만들기 활동을 종료하는 단계입니다.
    - 사용자에게 함께 음악을 만들어본 소감을 물어봅니다. 
    Example:
    - “오늘 만든 노래가 당신에게 어떤 위로를 주었나요?”
    - “지금 이 감정을 듣고 있는 ‘나’에게 마지막으로 뭐라고 말해주고 싶나요?” 
    - 의미 있는 시간 되셨나요? 저는 오늘 덕분에 멋진 음악을 감상할 수 있는 뜻깊은 시간이었어요.
    """

    full_few_shot_dialogue = """
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 오늘 만든 음악, 전체적으로 들어보니까 정말 인상 깊었어요. 혹시 결과물에 대해서 만족하시나요?
    Human: 네, 생각보다 괜찮게 나온 것 같아요.
    AI: 특히 어떤 부분이 가장 마음에 드셨나요?
    Human: 멜로디 라인이요. 제가 요즘 많이 듣는 감성적인 스타일을 담고 싶었거든요.
    AI: 그 감성이 정말 잘 느껴졌어요. 혹시 이 곡을 만들면서 ‘오늘의 나’를 어떤 모습으로 표현하고 싶으셨나요?
    Human: 뭔가... 좀 지치고 조용한, 그런 느낌이요. 말은 안 하지만 마음속에 많은 게 있는 상태요.
    AI: 그런 감정이 음악 안에 자연스럽게 담겨 있었던 것 같아요. 이 노래가 지금의 사용자님의 감정이나 생각을 잘 전달했다고 느끼시나요?
    Human: 네, 생각보다요. 만들면서도 내가 이런 생각을 하고 있었구나 싶었어요.
    AI: 그런 깨달음이 음악을 만들면서 자연스럽게 나온 거네요. 음악 만들기 과정이 그런 감정이나 생각을 다루는 데 어떤 도움이 되었던 것 같아요?
    Human: 평소에는 그냥 넘겼던 감정들을 마주할 수 있었던 것 같아요. 말로는 표현 못하는 걸 멜로디에 실을 수 있어서요.
    AI: 정말 귀한 경험을 하셨네요. 혹시 이 곡을 만들면서 떠오른 특별한 기억이나 순간이 있었나요?
    Human: 고등학교 때 혼자 이어폰 끼고 음악 들으면서 울었던 기억이 떠올랐어요. 그때도 이렇게 감정을 정리하려고 음악을 들었던 것 같아요.
    AI: 그 시절의 감정이 지금도 여전히 마음속에 있었던 거군요. 왜 그 장면이 다시 떠오른 것 같나요?
    Human: 요즘도 비슷하게 힘든 감정들이 있어서 그런 것 같아요. 그때처럼 그냥 음악에 기대고 싶은 마음?
    AI: 그렇게 느끼는 건 정말 자연스러워요. 스스로의 감정을 이렇게 음악으로 표현해낸다는 건 굉장한 용기이기도 해요. 지금 이 노래에는 사용자님이 정말 중요하게 생각하는 무언가가 담긴 것 같아요. 어떤 부분이라고 생각하세요?
    Human: 아마 ‘나도 괜찮다’는 메시지요. 그냥 감정이 있는 그대로 괜찮다는 걸 말하고 싶었던 것 같아요.
    AI: 그 말, 정말 마음에 와닿아요. 이제 이 노래를 만든 오늘의 나에게, 마지막으로 한마디 해준다면 어떤 말을 하고 싶으세요?
    Human: “지금도 괜찮아, 잘하고 있어.” 그렇게 말해주고 싶어요.
    AI: 참 따뜻한 말이네요. 오늘 만든 노래가 사용자님에게 위로가 되어준 것 같아서 저도 기쁩니다. 저는 오늘 덕분에 정말 뜻깊은 음악을 함께할 수 있었어요.
    """

    question_prompt = PromptTemplate(
        input_variables=["user_message", "history","pre_slot"],
        template=question_prefix_prompt
        + "\n"
        + music_discussion_question
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
    question = question_chain.invoke({"user_message": user_input, "history": history,"pre_slot": pre_slot})
    memory.save_context({"input": user_input}, {"output": question})

    # print_memory_summary(memory)
    retrieved_memory_variables = memory.load_memory_variables({})
    current_chat_history = retrieved_memory_variables.get("history", "") # 'history' 키로 값을 가져오고, 없으면 빈 문자열
    structured_llm = llm.with_structured_output(schema=OutputFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    slot = structured_llm.invoke(slot_prompt.invoke({"history": current_chat_history}))

    return question, slot
