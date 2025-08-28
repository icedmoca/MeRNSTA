import pytest
from storage.memory_log import MemoryLog
from config.settings import get_user_profile_id
import uuid

def test_fact_persistence_across_sessions(tmp_path):
    db_path = tmp_path / "test_memory.db"
    log = MemoryLog(str(db_path))
    session_id1 = str(uuid.uuid4())
    session_id2 = str(uuid.uuid4())
    user_profile_id = get_user_profile_id()
    triplet = ("favorite color", "is", "red", 1.0)
    log.store_triplets([triplet], session_id=session_id1, user_profile_id=user_profile_id)
    facts1 = log.semantic_search("favorite color", user_profile_id=user_profile_id)
    assert any(f.object == "red" for f in facts1)
    # New session, same profile
    log.store_triplets([("favorite food", "is", "sushi", 1.0)], session_id=session_id2, user_profile_id=user_profile_id)
    facts2 = log.semantic_search("favorite food", user_profile_id=user_profile_id)
    assert any(f.object == "sushi" for f in facts2)

def test_profile_based_search(tmp_path):
    db_path = tmp_path / "test_memory2.db"
    log = MemoryLog(str(db_path))
    session_id = str(uuid.uuid4())
    user_profile_id1 = "profileA"
    user_profile_id2 = "profileB"
    log.store_triplets([("favorite color", "is", "blue", 1.0)], session_id=session_id, user_profile_id=user_profile_id1)
    log.store_triplets([("favorite color", "is", "green", 1.0)], session_id=session_id, user_profile_id=user_profile_id2)
    factsA = log.semantic_search("favorite color", user_profile_id=user_profile_id1)
    factsB = log.semantic_search("favorite color", user_profile_id=user_profile_id2)
    assert any(f.object == "blue" for f in factsA)
    assert any(f.object == "green" for f in factsB)

def test_no_sentence_transformers():
    import sys
    assert 'sentence_transformers' not in sys.modules, 'sentence_transformers should not be imported' 