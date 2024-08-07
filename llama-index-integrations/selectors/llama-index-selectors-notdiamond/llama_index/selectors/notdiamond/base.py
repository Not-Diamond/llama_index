import logging
from typing import Sequence, List

from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from llama_index.selectors.base import (
    LLMSingleSelector,
    SelectorResult,
    SingleSelection,
)

from notdiamond import NotDiamond, LLMConfig, Metric

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


class NotDiamondSelectorResult(SelectorResult):
    def __init__(
        self, selections: Sequence[SingleSelection], session_id: str, *args, **kwargs
    ):
        super().__init__(selections, *args, **kwargs)
        self.session_id = session_id


class NotDiamondSelector(LLMSingleSelector):
    def __init__(
        self, metric: Metric = None, client: NotDiamond = None, *args, **kwargs
    ):
        # Not needed - we will route using our own client based on the query prompt
        _llm = None
        _prompt = None

        if metric and not isinstance(metric, Metric):
            raise ValueError(f"Invalid metric - needed type Metric but got {metric}")

        self._metric = metric or Metric("accuracy")
        self._client = client or NotDiamond()
        super().__init__(_llm, _prompt, *args, **kwargs)

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        llm_configs = self._choices_to_llm_configs(choices)
        messages = [{"role": "user", "content": query.query_str}]

        session_id, best_llm = self._client.model_select(
            messages=messages,
            llm_configs=llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or 5,
        )

        return _get_nd_selector_result(best_llm, llm_configs, session_id)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle, timeout: int = None
    ) -> SelectorResult:
        llm_configs = self._choices_to_llm_configs(choices)
        messages = [{"role": "user", "content": query.query_str}]

        session_id, best_llm = await self._client.amodel_select(
            messages=messages,
            llm_configs=llm_configs,
            metric=self._metric,
            notdiamond_api_key=self._client.api_key,
            max_model_depth=self._client.max_model_depth,
            hash_content=self._client.hash_content,
            tradeoff=self._client.tradeoff,
            preference_id=self._client.preference_id,
            tools=self._client.tools,
            timeout=timeout or 5,
        )

        return _get_nd_selector_result(best_llm, llm_configs, session_id)

    def _choices_to_llm_configs(
        self, choices: Sequence[ToolMetadata]
    ) -> List[LLMConfig]:
        llm_configs = []
        for choice in choices:
            if isinstance(choice, str):
                llm_configs.append(LLMConfig.from_string(choice))
            else:
                llm_configs.append(_tool_metadata_to_llm_config(choice))

        if self._client.llm_configs and set(llm_configs) != set(
            self._client.llm_configs
        ):
            LOGGER.warning(
                f"LLM configs do not match those in the NotDiamond client - will use configs passed to `aselect`."
            )
        return llm_configs


def _get_nd_selector_result(
    best_llm: LLMConfig, llm_configs: List[LLMConfig], session_id: str
) -> SelectorResult:
    """
    Given a LLMConfig returned by NotDiamond, build a SelectorResult.

    N.B. we inherit from SelectorResult to provide users with their request's session ID.
    """
    best_index = None
    for lcidx, llm_config in enumerate(llm_configs):
        if llm_config == best_llm:
            best_index = lcidx
            break
    if best_index is None:
        raise ValueError(
            f"Could not find best LLM config in the client - {best_llm}. This should not happen."
        )

    return NotDiamondSelectorResult(
        selections=[
            SingleSelection(
                index=best_index,
                reason=f"Not Diamond selected {best_llm} as best for this prompt.",
            )
        ],
        session_id=session_id,
    )


def _tool_metadata_to_llm_config(choice: ToolMetadata | str) -> LLMConfig:
    try:
        return LLMConfig.from_string(choice.description)
    except ValueError:
        pass

    try:
        return LLMConfig.from_string(choice.name)
    except ValueError:
        raise ValueError(
            f"Could not parse LLMConfig from provided tool metadata: {choice}. Need string of format '<provider>/<model>' as name or description."
        )
