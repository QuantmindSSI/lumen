import sqlite3

import pytest

from lumen._init import initialize_palace
from lumen.config import LumenConfig
from lumen.conversation import ConversationMemory
from lumen.search import SearchPipeline


@pytest.fixture
def temp_config(tmp_path):
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / ".lumen"),
    )


def test_initialize_palace_returns_correct_keys(temp_config):
    result = initialize_palace(temp_config)
    assert "conn" in result
    assert "embedder" in result
    assert "pipeline" in result
    assert "conversation" in result
    assert "tfc" in result
    assert "embedding_model_available" in result


def test_initialize_palace_conn_is_sqlite_connection(temp_config):
    result = initialize_palace(temp_config)
    assert isinstance(result["conn"], sqlite3.Connection)
    result["conn"].close()


def test_initialize_palace_creates_pipeline_and_conversation(temp_config):
    result = initialize_palace(temp_config)
    assert isinstance(result["pipeline"], SearchPipeline)
    assert isinstance(result["conversation"], ConversationMemory)
    result["conn"].close()


def test_initialize_palace_embedding_model_available_is_boolean(temp_config):
    result = initialize_palace(temp_config)
    assert isinstance(result["embedding_model_available"], bool)
    result["conn"].close()
