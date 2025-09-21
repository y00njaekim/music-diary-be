from llm_instance import llm
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#total_lyrics: full lyrics / change_lyrics: segment to edit / user_lyric_prompt: user instruction
def lyrics_change(total_lyrics: str, change_lyrics: str, user_lyric_prompt: str
    ) -> str:

    change_prefix_prompt="""
    [Lyrics Change Task]
    Rewrite the lyric segment according to the user's instructions.

    Segment to revise: {change_lyrics}
    User instruction: {user_lyric_prompt}

    [Guidelines]
    - Keep the edited segment as close in length as possible to the original.
    - Output only the updated lyric segment and nothing else.
    - Use the full lyrics below to maintain tone and flow.

    Full lyrics:
    {total_lyrics}
    """

    example_prompt = PromptTemplate.from_template(
    """
    Segment to revise: {change_lyrics}
    User instruction: {user_lyric_prompt}
    Full lyrics: {total_lyrics}
    
    -> Updated lyrics: {answer}
    """
    )

    examples = [
    {
        "change_lyrics": "Love always stays right here",
        "user_lyric_prompt": "Give it a lonelier and darker mood.",
        "total_lyrics": "[Intro] A quiet breeze seeps in, wrapping the deepest part of my heart, and the day that had stopped blooms again because of you. [Verse 1] At the end of a familiar alley, your laughter comes to mind and softly shakes me like a passing season. [Verse 2] Time flows indifferently, blooming flowers over scars, and before I know it your warmth remains in that place. [Chorus] Love always stays right here, shining on me like warm sunlight; you silently held me when I feared being forgotten. [Verse 3] On nights when I thought I was alone, I hear your breath, and the name that is you is tied to the deepest part of my heart. [Bridge] Even when we grow distant, I can reach you; if I gather the scattered dreams and walk along that light, I arrive at you in the end. [Chorus] Love always stays right here, warming even a lost heart as you quietly seep into me. [Outro] Even when the final season comes, you remain, lighting my day in the name of love.",
        "answer": "Love still lingers in this place"
    },
    {
        "change_lyrics":"Like a love that lost its way",
        "user_lyric_prompt":"Is there another expression instead of saying it lost its way?",
        "total_lyrics":"[Intro] In the stillness of a deep night, time has frozen, and this song begins with a heart no one knows. [Verse 1] Alone in the dark night, my heart feels lonely; like a love that lost its way, who could ever understand me? [Verse 2] Each day drags on like a year, and I can barely endure it as my heart cools in this endless solitude. [Chorus] Who will understand my heart, who will hold me close, who will know my heart in this endless love? [Verse 3] I find you in my dreams, yet you vanish again; without hope I long for the sky and call out your name. [Bridge] I cannot go on without you; only you can fill my heart, so find my smile at the edge of this sorrow. [Chorus] Who will understand my heart, who will hold me close, who will know my heart in this endless love? [Outro] Now it wraps around me quietly, whispering like the night sky, reaching for you even from afar as my final song.",
        "answer": "Like a wandering love"
    }

    ]

    change_suffix_prompt = """
    Segment to revise: {change_lyrics}
    User instruction: {user_lyric_prompt}
    Full lyrics: {total_lyrics}
    -> Updated lyrics:
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
