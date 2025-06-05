from llm_instance import llm
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#total_lyrics: 전체가사 / change_lyrics: 바꿀가사부분 / user_lyric_prompt: 어떻게 바꿀지에 대한 지시
def lyrics_change(total_lyrics: str, change_lyrics: str, user_lyric_prompt: str
    ) -> str:

    change_prefix_prompt="""
    [Lyrics Change Task]
    사용자가 요구하는 지시에 따라 가사를 변경하세요.

    변경할 가사 부분: {change_lyrics}
    사용자가 요구하는 지시: {user_lyric_prompt}

    [유의사항]
    - 되도록 변경할 가사 부분과 글자수를 맞추어 변경하세요.
    - 오로지 변경한 가사 부분의 텍스트만 output으로 출력합니다.
    - 전체 가사는 다음과 같습니다. 어색하지 않게 생성하세요. 

    전체가사:
    {total_lyrics}
    """

    example_prompt = PromptTemplate.from_template(
    """
    변경할 가사 부분: {change_lyrics}
    사용자가 요구하는 지시: {user_lyric_prompt}
    전체 가사: {total_lyrics}
    
    → 변경된 가사: {answer}
    """
    )

    examples = [
    {
        "change_lyrics": "사랑은 언제나 그 자리에",
        "user_lyric_prompt": "좀 더 쓸쓸하고 어두운 분위기로 바꿔줘",
        "total_lyrics": "[Intro] 조용히 스며든 바람 내 맘 깊은 곳을 감싸 멈춰있던 나의 하루 너로 인해 다시 피어나 [Verse 1] 익숙한 골목 끝에서 네 웃음이 떠오르고 지나간 계절처럼 아련하게 나를 흔들어 [Verse 2] 시간은 무심히 흘러 상처 위에 꽃을 피워 어느새 그 자리에 너의 온기가 머물러 [Chorus] 사랑은 언제나 그 자리에 햇살처럼 따스하게 나를 비춰줘 잊혀질까 두려운 나를 말없이 안아주는 너였어 [Verse 3] 혼자인 줄 알았던 밤 너의 숨결이 들려와 내 맘 속 가장 깊은 곳 그대란 이름이 맺혀 있어 [Bridge] 멀어져도 닿을 수 있어 흩어진 꿈을 모아 그 빛 따라 걸어가면 결국 너에게 도착해 [Chorus] 사랑은 언제나 그 자리에 햇살처럼 따스하게 나를 비춰줘 길 잃은 마음마저 감싸 조용히 내게 스며든 너 [Outro] 마지막 계절이 와도 너는 그대로 머물러 사랑이란 그 이름으로 내 하루를 비춰줘",
        "answer": "사랑은 아직도 그곳에 머물러 있네"
    },
    {
        "change_lyrics":"길 잃은 사랑처럼",
        "user_lyric_prompt":"길 잃은 이라는 표현말고 다른 표현 없을까?",
        "total_lyrics":"[Intro] 깊은 밤의 적막 속에 멈춰버린 시간처럼 누구도 모르는 내 맘 이 노래로 시작해 [Verse 1] 어두운 밤 혼자서 나의 마음 외로워 길 잃은 사랑처럼 누가 내 마음을 알까 [Verse 2] 하루가 일년처럼 매일 견딜 수 없어 끝없는 이 고독 속에 나의 마음은 식어져가 [Chorus] 누가 내 마음을 알까 누가 나를 안아줄까 끝없는 사랑 속에 내 마음을 누가 알까 [Verse 3] 꿈속에서 찾은 너 하지만 또 사라져 희망 없이 하늘만 그리워 너를 부른다 [Bridge] 그대 없이 난 안돼 그대만이 나의 마음 채워줘 이 슬픔의 끝자락에 나의 웃음을 찾아줘 [Chorus] 누가 내 마음을 알까 누가 나를 안아줄까 끝없는 사랑 속에 내 마음을 누가 알까 [Outro] 이젠 조용히 감싸와 밤하늘에 속삭이듯 멀리 있어도 닿기를 내 마지막 노래로",
        "answer": "방황하는 사랑처럼"
    }

    ]

    change_suffix_prompt = """
    변경할 가사 부분: {change_lyrics}
    사용자가 요구하는 지시: {user_lyric_prompt}
    전체 가사: {total_lyrics}
    → 변경된 가사:
    """

    change_prompt = FewShotPromptTemplate(
    prefix=change_prefix_prompt,
    suffix=change_suffix_prompt,
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["change_lyrics", "user_lyric_prompt", "total_lyrics"]
    )

    change_lyrics_chain = change_prompt | llm | StrOutputParser()
    after_lyrics = change_lyrics_chain.invoke({"total_lyrics": total_lyrics, "change_lyrics": change_lyrics, "user_lyric_prompt": user_lyric_prompt})

    return after_lyrics