"""Unit tests for the LangChain implementation of /rags endpoints."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException, status
from pytest_mock import MockerFixture

from app.endpoints.rags_langchain import get_rag_langchain, list_rags_langchain
from configuration import AppConfig


def _make_byok_config(tmp_path: Any) -> AppConfig:
    """Create an AppConfig with BYOK RAG entries for testing."""
    db_file = Path(tmp_path) / "test.db"
    db_file.touch()
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "langchain": {
                "providers": {
                    "openai": {
                        "api_key": "test-key",
                        "models": ["gpt-4"],
                    }
                },
                "default_provider": "openai",
                "default_model": "gpt-4",
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "authorization": {"access_rules": []},
            "byok_rag": [
                {
                    "rag_id": "ocp-4.18-docs",
                    "rag_type": "inline::faiss",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "vector_db_id": "vs_abc123",
                    "db_path": str(db_file),
                },
                {
                    "rag_id": "company-kb",
                    "rag_type": "inline::faiss",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "vector_db_id": "vs_def456",
                    "db_path": str(db_file),
                },
            ],
        }
    )
    return cfg


@pytest.mark.asyncio
async def test_list_rags_langchain_returns_configured_rags(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test that list_rags_langchain returns RAGs from configuration."""
    byok_config = _make_byok_config(str(tmp_path))
    mocker.patch("app.endpoints.rags_langchain.configuration", byok_config)

    response = await list_rags_langchain()
    assert len(response.rags) == 2
    assert "ocp-4.18-docs" in response.rags
    assert "company-kb" in response.rags


@pytest.mark.asyncio
async def test_list_rags_langchain_empty_config(mocker: MockerFixture) -> None:
    """Test that list_rags_langchain returns empty list when no RAGs configured."""
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "langchain": {
                "providers": {
                    "openai": {
                        "api_key": "test-key",
                        "models": ["gpt-4"],
                    }
                },
                "default_provider": "openai",
                "default_model": "gpt-4",
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "authorization": {"access_rules": []},
            "byok_rag": [],
        }
    )
    mocker.patch("app.endpoints.rags_langchain.configuration", cfg)

    response = await list_rags_langchain()
    assert len(response.rags) == 0


@pytest.mark.asyncio
async def test_get_rag_langchain_success(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test that get_rag_langchain returns RAG info from configuration."""
    byok_config = _make_byok_config(str(tmp_path))
    mocker.patch("app.endpoints.rags_langchain.configuration", byok_config)

    response = await get_rag_langchain("ocp-4.18-docs")
    assert response.id == "ocp-4.18-docs"
    assert response.name == "ocp-4.18-docs"
    assert response.status == "active"
    assert response.object == "vector_store"


@pytest.mark.asyncio
async def test_get_rag_langchain_not_found(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test that get_rag_langchain returns HTTP 404 when RAG not found."""
    byok_config = _make_byok_config(str(tmp_path))
    mocker.patch("app.endpoints.rags_langchain.configuration", byok_config)

    with pytest.raises(HTTPException) as e:
        await get_rag_langchain("nonexistent-rag")
    assert e.value.status_code == status.HTTP_404_NOT_FOUND
    detail = e.value.detail
    assert isinstance(detail, dict)
    assert "response" in detail
    assert "Rag not found" in detail["response"]  # type: ignore[index]
