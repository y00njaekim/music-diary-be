from llm_instance import llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def termination(total_dialogue: str, llm, pre_slot) -> str:
    termination_question="""
    [Termination Task]
    - Thank the user for today and close the session warmly.
    - Offer a farewell instead of asking another question.
    Examples:
    - "I truly enjoyed our time together today. Let's meet again soon."
    - "Thank you for sharing; this was a meaningful time for me."
    - "I will be right here whenever you would like to return."
    """

    question_prompt = PromptTemplate(template=termination_question)
    question_chain = question_prompt | llm | StrOutputParser()
    question = question_chain.invoke({})

    history_summary_prompt="""
    [Dialogue Summary Task]
    - Summarize the conversation in 3-4 sentences.
    - Focus on the user's challenges, the musical concept, and key musical details.

    Examples:
    - Based on feelings of "lethargy and loneliness," we created lyrics and music about understanding and comforting the self who is alone.
    - Grounded in the anxiety of "how others see me," we made a piece about accepting myself as I am.
    - Inspired by the struggle of "emptiness in a repetitive routine," we composed lyrics and music about finding meaning in small things.
    [Guidelines]
    - Write 3-4 sentences (plain text, no bullet points).
    - Use simple words so the summary is easy to read for someone with limited hearing and literacy.
    - This conversation comes from a music-therapy songwriting session; skip repeated boilerplate statements between sessions.

    """
    summary_prompt = PromptTemplate(
        input_variables=["total_dialogue","pre_slot"],
        template=history_summary_prompt
        + "\n"
        + "Generate the response using the references below."
        + "Previous Slot: {pre_slot}\n"
        + "Chat history: {total_dialogue}\n"
    )

    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"total_dialogue": total_dialogue,"pre_slot": pre_slot })
    

    response = summary + '\n' + question + '\n' + 'Please complete the survey using the link below.' + '\n' + 'https://docs.google.com/forms/u/0/d/1a_huYPVyQjLC-VOHZ5zzjpbayVjhuc55QAWg1tsFw6w/preview'
    
    return response, summary
