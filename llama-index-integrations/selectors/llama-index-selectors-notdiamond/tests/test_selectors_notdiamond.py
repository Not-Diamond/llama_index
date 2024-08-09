import os
import pytest
import random
from typing import List
from unittest.mock import MagicMock
import uuid

from llama_index.core.tools import ToolMetadata
from llama_index.selectors.notdiamond.base import (
    _get_nd_selector_result,
    NotDiamondSelector,
)

from notdiamond import LLMConfig


def test_get_nd_selector_result():
    llm_configs = [
        LLMConfig(provider="openai", model="gpt-4o"),
        LLMConfig(provider="openai", model="gpt-3.5-turbo"),
        LLMConfig(provider="anthropic", model="claude-3-opus-20240229"),
    ]
    best_llm = llm_configs[0]
    session_id = str(uuid.uuid4())
    result = _get_nd_selector_result(best_llm, llm_configs, session_id)
    assert result.index == 0

    random.shuffle(llm_configs)
    actual_idx = llm_configs.index(best_llm)
    result = _get_nd_selector_result(best_llm, llm_configs, session_id)
    assert result.index == actual_idx


@pytest.fixture()
def nd_selector():
    from notdiamond import NotDiamond

    os.environ["OPENAI_API_KEY"] = "test"
    os.environ["ANTHROPIC_API_KEY"] = "test"

    _client = MagicMock(wraps=NotDiamond(api_key="test"))
    return NotDiamondSelector(client=_client)


class TestNotDiamondSelector:
    choices: List[ToolMetadata] = [
        ToolMetadata(name="foobar", description="anthropic/claude-3-opus-20240229"),
        ToolMetadata(name="foobar", description="openai/gpt-4o"),
    ]

    def test_select(self, nd_selector):
        prompt = "Please describe the llama_index framework in 280 characters or less."
        session_id = str(uuid.uuid4())
        nd_selector._client.model_select.return_value = (
            session_id,
            LLMConfig.from_string(self.choices[0].description),
        )
        result = nd_selector.select(self.choices, prompt)
        assert result.index == 0

    @pytest.mark.asyncio()
    async def test_aselect(self, nd_selector):
        async def aselect(*args, **kwargs):
            return (session_id, LLMConfig.from_string(self.choices[0].description))

        prompt = "Please describe the llama_index framework in 280 characters or less."
        session_id = str(uuid.uuid4())
        nd_selector._client.amodel_select = aselect
        result = await nd_selector.aselect(self.choices, prompt)
        assert result.index == 0
