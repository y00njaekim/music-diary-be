import os
from enum import Enum
from typing import Optional

from supabase import create_client, Client

from chatbot.types import CombinedSlot


class SEARCH_OPTION(Enum):
    ALL = 0
    ID = 1
    LAST_K = 2
    LATEST = 3


REFERENCE_RULE = {
    "diary": "user_id",
    "chat": "session_id",
    "keywords": "session_id",
    "lyrics": "session_id",
    "state": "session_id",
    "summary": "session_id",
    "music": "lyrics_id",
    "musicVis": "music_id",
}


class DBManager:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)

    def _insert(self, table: str, data: dict):
        response = self.supabase.table(table).insert(data).execute()
        return response

    def _search(self, table: str, data: dict):
        query = self.supabase.table(table).select("*")

        for column, value in data.items():
            if column == "n":
                continue
            query = query.eq(column, value)

        # if searching condition is 'the latest' or 'last-k'
        if "n" in data:
            query = query.order("created_at", desc=True).limit(data["n"])

        return query.execute()

    def insert_diary(self, user_id: str):
        return self._insert("diary", {"user_id": user_id})

    def insert_chat(self, session_id: str, text: str, user_say: bool):
        return self._insert("chat", {"session_id": session_id, "text": text, "user_says": user_say})

    def insert_state(self, session_id: str, state_name: str):
        return self._insert("state", {"session_id": session_id, "state_name": state_name})

    def insert_keywords(self, session_id: str, keywords: CombinedSlot):
        return self._insert(
            "keywords",
            {
                "session_id": session_id,
                "keywords": keywords,
            },
        )

    def insert_lyrics(self, session_id: str, chat_id: Optional[str], lyrics: str):
        return self._insert(
            "lyrics",
            {
                "session_id": session_id,
                "chat_id": chat_id,
                "lyrics": lyrics,
            },
        )

    def insert_music(self, lyrics_id: str, prompt: str, url: str, title: str):
        return self._insert("music", {"lyrics_id": lyrics_id, "prompt": prompt, "url": url, "title": title})

    def insert_music_vis(self, music_id: str, vis_data: dict):
        return self._insert("musicVis", {"music_id": music_id, "vis_data": vis_data})

    def insert_summary(self, session_id: str, summary: str, latest_chat: str, latest_music: Optional[str], latest_state: str, latest_keywords: str):
        return self._insert(
            "summary",
            {
                "session_id": session_id,
                "summary": summary,
                "latest_chat": latest_chat,
                "latest_music": latest_music,
                "latest_state": latest_state,
                "latest_keywords": latest_keywords,
            },
        )

    def search_latest_summary(self, user_id: str):
        """
        user_id를 기반으로 가장 최근의 요약(summary)을 가져옵니다.
        summary와 diary 테이블을 inner join하여 조회합니다.
        """
        return (
            self.supabase.table("summary")
            .select("summary, diary!inner(*)")
            .eq("diary.user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .maybe_single()
            .execute()
        )

    def search_summaries_by_user(self, user_id: str):
        """
        user_id를 기반으로 모든 요약(summary)을 가져옵니다.
        summary와 diary 테이블을 inner join하여 조회합니다.
        """
        # `summary` 테이블에서 `summary_id`, `summary`, `created_at`를 선택하고,
        # `diary` 테이블과 내부 조인하여 `user_id`로 필터링합니다.
        # 생성일(`created_at`)을 기준으로 내림차순 정렬합니다.
        response = (
            self.supabase.table("summary")
            .select("summary_id, summary, created_at, diary!inner(user_id)")
            .eq("diary.user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response

    def search_music_details_by_summary(self, summary_id: str, user_id: str):
        """
        summary_id와 user_id를 기반으로 연결된 음악(music)과 시각화 데이터(musicVis)를 가져옵니다.
        """
        response = (
            self.supabase.table("summary")
            .select("latest_music(url, musicVis(vis_data)), diary!inner(user_id)")
            .eq("summary_id", summary_id)
            .eq("diary.user_id", user_id)
            .single()
            .execute()
        )
        return response

    def search(self, table: str, reference: str, ref_id: str, search_option: str, **kwargs: dict):
        """
        kwargs:
            id: for search by id
            n: for search by recent n items
        """

        if reference is not REFERENCE_RULE[table]:
            raise ValueError(f"{table} query must refer to '{REFERENCE_RULE[table]}'")

        data = {reference: ref_id}

        if search_option == SEARCH_OPTION.ALL:
            pass
        elif search_option == SEARCH_OPTION.ID:
            if "id" not in kwargs:
                raise ValueError("Need the argument 'session_id' to query by id")
            if table == "diary":
                data[f"session_id"] = kwargs["id"]
            else:
                data[f"{table}_id"] = kwargs["id"]
        elif search_option == SEARCH_OPTION.LAST_K:
            if "n" not in kwargs:
                raise ValueError("Need the argument 'n' to query the last-n items")
            data["n"] = kwargs["n"]
        else:  # Latest
            data["n"] = 1

        return self._search(table, data)

    # def update_current_info(self, user_id: str, session_id: str):
    #     self.current_user_id = user_id
    #     self.current_session_id = session_id

    # def get_current_user(self):
    #     return self.current_user_id

    # def get_current_session(self):
    #     return self.current_session_id
