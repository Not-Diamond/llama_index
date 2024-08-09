import logging
import os
from typing import Sequence

from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.base.base_selector import SelectorResult
from llama_index.core.selectors import LLMSingleSelector

from notdiamond import NotDiamond, LLMConfig, Metric

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


class NotDiamondSelectorResult(SelectorResult):
    """A single selection of a choice provided by Not Diamond."""

    session_id: str

    @classmethod
    def from_selector_result(
        cls, selector_result: SelectorResult, session_id: str
    ) -> "NotDiamondSelectorResult":
        return cls(session_id=session_id, **selector_result.dict())


class NotDiamondSelector(LLMSingleSelector):
    def __init__(
        self,
        client: NotDiamond,
        metric: Metric = None,
        timeout: int = 10,
        api_key: str = None,
        *args,
        **kwargs,
    ):
        # Not needed - we will route using our own client based on the query prompt
        # Add @property for _llm here
        _encap_selector = LLMSingleSelector.from_defaults()
        self._prompt = _encap_selector._prompt

        api_key = api_key or os.getenv("NOTDIAMOND_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided and NOTDIAMOND_API_KEY not found in environment."
            )

        if getattr(client, "llm_configs", None):
            raise ValueError(
                "NotDiamond client must have llm_configs before creating a NotDiamondSelector."
            )

        if metric and not isinstance(metric, Metric):
            raise ValueError(f"Invalid metric - needed type Metric but got {metric}")
        self._metric = metric or Metric("accuracy")

        self._client = client
        self._llms = [
            self._llm_config_to_client(llm_config)
            for llm_config in self._client.llm_configs
        ]
        self._timeout = timeout
        super().__init__(_encap_selector._llm, _encap_selector._prompt, *args, **kwargs)

    def _llm_config_to_client(self, llm_config: LLMConfig | str) -> LLM:
        if isinstance(llm_config, str):
            llm_config = LLMConfig.from_string(llm_config)
        provider, model = llm_config.provider, llm_config.model

        output = None
        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            output = OpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic

            output = Anthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "cohere":
            from llama_index.llms.cohere import Cohere

            output = Cohere(model=model, api_key=os.getenv("COHERE_API_KEY"))
        elif provider == "mistral":
            from llama_index.llms.mistralai import MistralAI

            output = MistralAI(model=model, api_key=os.getenv("MISTRALAI_API_KEY"))
        elif provider == "togetherai":
            from llama_index.llms.together import TogetherLLM

            output = TogetherLLM(model=model, api_key=os.getenv("TOGETHERAI_API_KEY"))
        else:
            raise ValueError(
                f"Unsupported provider for NotDiamond llama_index integration: {provider}"
            )

        return output

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        """
        Call Not Diamond to select the best LLM for the given prompt, then have the LLM select the best tool.
        """
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": query.query_str},
        ]

        session_id, best_llm = self._client.model_select(
            messages=messages,
            llm_configs=self._client.llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or self._timeout,
        )

        self._llm = self._llm_config_to_client(best_llm)

        return NotDiamondSelectorResult.from_selector_result(
            super()._select(choices, query), session_id
        )

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        """
        Call Not Diamond to select the best LLM for the given prompt, then have the LLM select the best tool.
        """
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": query.query_str},
        ]

        session_id, best_llm = await self._client.amodel_select(
            messages=messages,
            llm_configs=self._client.llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or self._timeout,
        )

        self._llm = self._llm_config_to_client(best_llm)

        return NotDiamondSelectorResult.from_selector_result(
            super()._select(choices, query), session_id
        )
