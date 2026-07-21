"""Tests for lumen.integrations.mcp_server."""

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    pytest.importorskip("mcp", reason="mcp package not installed") is None,
    reason="mcp package not installed",
)

from lumen.integrations.mcp_server import mcp, app_lifespan  # noqa: E402


class TestMCPServer:
    def test_tools_registered(self):
        async def _list():
            tools = await mcp.list_tools()
            return [t.name for t in tools]

        tool_names = asyncio.run(_list())
        assert "lumen_search" in tool_names
        assert "lumen_store" in tool_names
        assert "lumen_assemble" in tool_names
        assert "lumen_turn" in tool_names
        assert "lumen_feedback" in tool_names
        assert "lumen_status" in tool_names

    def test_lumen_store_and_search(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        models = tmp_path / "models"
        store.mkdir()
        models.mkdir()
        monkeypatch.setenv("LUMEN_STORE_PATH", str(store))
        monkeypatch.setenv("LUMEN_MODEL_PATH", str(models))

        async def _run():
            async with app_lifespan(mcp):
                store_res = await mcp.call_tool(
                    "lumen_store",
                    {
                        "content": "Use SQLite for the cache layer because it is zero-config.",
                        "room": "decisions",
                        "source_type": "user_input",
                    },
                )
                store_text = store_res[0][0].text if store_res and store_res[0] else ""
                assert "Stored chunk_id=" in store_text

                search_res = await mcp.call_tool(
                    "lumen_search",
                    {"query": "SQLite cache", "top_k": 5},
                )
                search_text = search_res[0][0].text if search_res and search_res[0] else ""
                assert "SQLite" in search_text or "No memories found" in search_text

        asyncio.run(_run())

    def test_lumen_status(self, tmp_path, monkeypatch):
        store = tmp_path / "store"
        models = tmp_path / "models"
        store.mkdir()
        models.mkdir()
        monkeypatch.setenv("LUMEN_STORE_PATH", str(store))
        monkeypatch.setenv("LUMEN_MODEL_PATH", str(models))

        async def _run():
            async with app_lifespan(mcp):
                res = await mcp.call_tool("lumen_status", {})
                text = res[0][0].text if res and res[0] else ""
                assert "Lumen Status" in text
                assert "Rooms:" in text

        asyncio.run(_run())
