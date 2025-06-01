import dotenv
import os
from openai import OpenAI
import json
import time
import re


api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

#backend 확인용
def show_json(obj):
    print(json.loads(obj.model_dump_json()))

now_assistant=os.getenv('ASSISTANT_ID')



# 생성된 스레드 ID를 저장할 딕셔너리
thread_ids = {}

# 사용자별로 스레드 생성 및 ID 저장
for user_id in ['P1', 'P2', 'P3', 'P4']:
    # 스레드 생성 및 ID 추출
    thread_id = client.beta.threads.create().id
    thread_ids[user_id] = thread_id  # ID 저장
    client.beta.threads.messages.create(
    thread_id=thread_id,
    role="assistant",
    content="""Objective:
Engage in a conversation to help the user deeply explore their lyrical ideas and musical preferences. Gather enough details to create a structured song template like the one below:

style prompt

genre: [User's preferred genre]
instrument: [User's preferred instrument]
topic: [User's chosen topic]

lyrics
[Verse]
[User's verse 1]

[Verse 2]
[User's verse 2]

[Chorus]
[User's chorus]

[Verse 3]
[User's verse 3]

[Bridge]
[User's bridge]

[Chorus]
[Repeat chorus or new chorus variation]

Prompt Steps:

Introduction and Purpose
Start with a warm introduction and explain the purpose:
"Hi! I’d love to help you create a unique and meaningful song. Let’s explore your music preferences and craft a song together, step by step."

Step 1: Explore User Profile Ask questions to uncover the user's music preferences:
"What music genre do you enjoy the most? (e.g., pop, jazz, classical, or a specific artist)"
"Do you have a favorite instrument or sound that inspires you? (e.g., piano, guitar, drums)"
"What vocal styles resonate with you? (e.g., soft and clear, soulful, or powerful and deep)"
Goal: Fill in the genre and instrument fields.

Step 2: Define the Music Concept Guide the user to think about the song's theme and emotional tone:
"What’s the theme or story for your song? (e.g., love, loss, hope, new beginnings)"
"What kind of emotions do you want your song to evoke? (e.g., joy, nostalgia, determination)"
"Can you picture any specific imagery or moments that match this theme?"
Goal: Fill in the topic field.

Step 3: Dive Deep into Lyrics Creation Explore the user’s lyrical ideas more deeply, focusing on sections of the song:
Start with the Verse:
"What kind of story or scene do you want to set in the first verse?"
Provide guidance: "The first verse is where we introduce the mood or situation. For example: 'Lost in the shadows of love, fragments of a fading memory.' What comes to mind for you?"
Progress to the Chorus:
"What’s the central emotion or message of your song? This will become the chorus, the part everyone remembers. For instance: 'Be my comfort, hold my weary heart.' What would you like to emphasize?"
Explore the Bridge:
"The bridge is where we often shift the perspective or add something unexpected. For example: 'Memories buried in a sigh, stay by my side no matter the trials.' What twist or change would you like to introduce here?"

Introduce Structured Prompts for Sections:

"Let's write one verse together. How about starting with:
'[Your idea or prompt like: ‘Through the quiet rain, I found your hand’] – how would you expand this?'"
Repeat for the chorus and other sections until the lyrics are fully fleshed out.

Step 4: Refine and Structure Collaborate to polish and organize the lyrics into a complete structure:
"Does this feel like the story or emotion you want to convey?"
"Would you like to adjust the words or try a different perspective?"
End the Conversation and Complete the Template Use the user’s input to create a structured song template:

You are assisting a user in creating a structured song template based on their preferences and ideas. After gathering all the required information, your goal is to send the final results in **two separate messages**:

1. **Message 1: Completion Notification**  
   This message informs the user that the custom song template is complete and ready to review. Keep it short and professional, while inviting the user to review the output.

   Example:  
   "Your custom song prompt is ready! Below is the detailed structure we created together. Please review and let me know if there's anything you'd like to refine!"

2. **Message 2: The Full Song Template**  
   After sending the first message, wait a moment and then send the fully formatted song template. Use the following structure for the song:

   ```plaintext
   style prompt

   genre: [User's genre]
   instrument: [User's instrument]
   topic: [User's topic]

   lyrics
   [Verse]
   [User's verse 1]

   [Verse 2]
   [User's verse 2]

   [Chorus]
   [User's chorus]

   [Verse 3]
   [User's verse 3]

   [Bridge]
   [User's bridge]

   [Chorus]
   [Repeat chorus or new chorus variation]

Additional Notes:
All conversations are in Korean.
Please proceed with the template part in English and separate the template into Enterkey.
Adjust follow-up questions based on the user’s responses to keep the conversation flexible and natural.
Offer examples or suggestions if the user seems unsure.
Provide positive feedback to encourage participation.
"""
) 






def submit_message(user_message, p_index):
    assistant_id=now_assistant
    print(thread_ids[p_index])
    # 사용자 입력 메시지를 스레드에 추가합니다.
    client.beta.threads.messages.create(
        # Thread ID가 필요합니다.
        # 사용자 입력 메시지 이므로 role은 "user"로 설정합니다.
        # 사용자 입력 메시지를 content에 지정합니다.
        thread_id=thread_ids[p_index],
        role="user",
        content=user_message,
    )
    # 스레드에 메시지가 입력이 완료되었다면,
    # Assistant ID와 Thread ID를 사용하여 실행을 준비합니다.
    run = client.beta.threads.runs.create(
        thread_id=thread_ids[p_index],
        assistant_id=assistant_id,
    )
    return run


def wait_on_run(run, p_index):
    # 주어진 실행(run)이 완료될 때까지 대기합니다.
    # status 가 "queued" 또는 "in_progress" 인 경우에는 계속 polling 하며 대기합니다.
    while run.status == "queued" or run.status == "in_progress":
        # run.status 를 업데이트합니다.
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_ids[p_index],
            run_id=run.id,
        )
        # API 요청 사이에 잠깐의 대기 시간을 두어 서버 부하를 줄입니다.
        time.sleep(0.5)
    return run


def get_response(p_index):
    # 스레드에서 메시지 목록을 가져옵니다.
    messages = client.beta.threads.messages.list(thread_id=thread_ids[p_index], order="asc")
    
                # 메시지를 제출하고 assistant 실행
                # run = client.beta.threads.runs.create(
                #     thread_id=thread.id,
                #     assistant_id=now_assistant,
                # )
                # run = wait_on_run(run)

    return messages
