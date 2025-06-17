import pytest
from unittest.mock import patch, MagicMock
import os
import sys
from dotenv import load_dotenv
import uuid

# src 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from database.manager import DBManager, SEARCH_OPTION

FAKE_SUPABASE_URL = "http://test.supabase.co"
FAKE_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"


@pytest.fixture
def fake_db_manager():
    with patch("database.manager.create_client") as mock_create_client:
        with patch.dict("os.environ", {"SUPABASE_URL": FAKE_SUPABASE_URL, "SUPABASE_KEY": FAKE_SUPABASE_KEY}):

            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            # DBManager의 self.supabase에 mock_client가 할당
            manager = DBManager()
            yield manager


class TestDBManager:
    def test_initialization(self, fake_db_manager: DBManager):
        assert fake_db_manager is not None
        assert fake_db_manager.supabase is not None

        # DBManager의 supabase 가 모킹된 객체인지 확인
        assert isinstance(fake_db_manager.supabase, MagicMock)

    def test_insert_diary(self, fake_db_manager: DBManager):

        user_id = "test_user_id"
        fake_db_manager._insert = MagicMock()
        fake_db_manager.insert_diary(user_id)
        fake_db_manager._insert.assert_called_once_with("diary", {"user_id": user_id})  # Table, {reference_key: value}

    def test_search_with_invalid_reference(self, fake_db_manager: DBManager):
        with pytest.raises(ValueError) as excinfo:
            fake_db_manager.search(table="diary", reference="invalid_reference_key", ref_id="invalid_reference_value", search_option=SEARCH_OPTION.ALL)
        assert "diary query must refer to 'user_id'" in str(excinfo.value)

    def test_search_last_k_without_n(self, fake_db_manager: DBManager):
        with pytest.raises(ValueError) as excinfo:
            fake_db_manager.search(table="chat", reference="session_id", ref_id="some_value", search_option=SEARCH_OPTION.LAST_K)
        assert "Need the argument 'n' to query the last-n items" in str(excinfo.value)


# --- 통합 테스트용 코드 추가 ---


@pytest.fixture(scope="module")
def real_db_manager():
    """
    모듈 단위로 실행되는 픽스쳐.
    테스트 시작 시 .env.dev 파일을 로드하고 실제 DBManager 인스턴스를 생성.
    (데이터 정리는 각 테스트별 픽스쳐에서 담당)
    """
    # .env.dev 파일의 정확한 상대 경로를 지정
    env_path = os.path.join(os.path.dirname(__file__), "../../src/.env.dev")
    load_dotenv(dotenv_path=env_path)

    # 환경변수 로드 확인
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        pytest.skip("TEST DB 정보가 .env.dev에 없습니다. 통합 테스트를 건너뜁니다.")

    manager = DBManager()
    yield manager


@pytest.fixture
def diary_entry(real_db_manager: DBManager):
    """
    테스트 함수를 위한 diary 항목을 생성하고, 테스트가 끝나면 정리하는 픽스쳐.
    """
    user_id = "fe54aae6-0ad0-4e10-bf1b-fe480d9fc41a"
    insert_response = real_db_manager.insert_diary(user_id=user_id)

    assert len(insert_response.data) > 0, "Diary entry 생성에 실패했습니다."
    session_id = insert_response.data[0]["session_id"]
    print(f"[Setup] Diary entry created with session_id: {session_id}")

    # 테스트 함수에 DBManager와 생성된 ID들을 전달
    yield real_db_manager, user_id, session_id

    # --- 테스트 함수 종료 후 데이터 정리 ---
    print(f"\n[Teardown] Cleaning up data for session: {session_id}")
    try:
        # 종속된 chat 데이터부터 삭제
        real_db_manager.supabase.table("chat").delete().eq("session_id", session_id).execute()
        # diary 데이터 삭제
        real_db_manager.supabase.table("diary").delete().eq("user_id", user_id).execute()
        print("[Teardown] Test data cleaned up successfully.")
    except Exception as e:
        print(f"[Teardown] Error during cleanup: {e}")


@pytest.mark.integration
class TestDBManagerIntegration:
    def test_insert_and_search_chat(self, diary_entry):
        """
        실제 테스트 DB에 chat 데이터를 삽입하고 검색하여 확인하는 테스트
        """
        # diary_entry 픽스쳐로부터 필요한 값들을 받아옴
        real_db_manager, user_id, session_id = diary_entry

        # 10개의 chat 데이터를 삽입
        for i in range(10):
            test_text = f"test_text_{i}"
            test_user_say = i % 2 == 0
            insert_response = real_db_manager.insert_chat(session_id=session_id, text=test_text, user_say=test_user_say)

            # 삽입 결과 확인
            assert len(insert_response.data) == 1, "Chat 데이터가 정상적으로 삽입되지 않았습니다."
            inserted_chat = insert_response.data[0]
            assert inserted_chat["session_id"] == session_id

        # 삽입된 chat 데이터를 검색
        search_response = real_db_manager.search(table="chat", reference="session_id", ref_id=session_id, search_option=SEARCH_OPTION.ALL)

        # 검색 결과 검증
        assert len(search_response.data) == 10, f"SEARCH_OPTION.ALL이 모든 데이터를 가져오지 못했습니다. (결과: {len(search_response.data)}개)"
        print(f"\nSEARCH_OPTION.ALL 결과 (총 {len(search_response.data)}개):")
        for item in search_response.data:
            print(f"  - {item['text']} (생성 시각: {item['created_at']})")
