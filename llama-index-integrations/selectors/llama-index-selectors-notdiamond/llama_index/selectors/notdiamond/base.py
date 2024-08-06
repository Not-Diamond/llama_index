from typing import Sequence

from llama_index.core.schema import QueryBundle
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from llama_index.core.selectors.prompts import MultiSelectPrompt
from llama_index.core.tools.types import ToolMetadata
from llama_index.selectors.base import (
    LLMMultiSelector,
    SelectorResult,
    _build_choices_text,
    _structured_output_to_selector_result,
)


class NotDiamondSelector(LLMMultiSelector):
    def __init__(
        self, llm: LLMPredictorType, prompt: MultiSelectPrompt, *args, **kwargs
    ):
        super().__init__(llm, *args, **kwargs)

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        pass

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        context_list = _build_choices_text(choices)
        max_outputs = self._max_outputs or len(choices)

        prediction = await self._llm.apredict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_outputs=max_outputs,
            context_list=context_list,
            query_str=query.query_str,
        )

        assert self._prompt.output_parser is not None
        parsed = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parsed)
