import logging

from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.selectors.notdiamond import (
    NotDiamondSelector,
    _tool_metadata_to_llm_config,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


class NotDiamondQueryEngine(RouterQueryEngine):
    def __init__(self, *args, **kwargs) -> None:
        warn_msg = "NotDiamondQueryEngine does not support configurable selector. Will use NotDiamondSelector by default."
        if "selector" in kwargs and not isinstance(
            kwargs["selector"], NotDiamondSelector
        ):
            LOGGER.warning(warn_msg)
            kwargs["selector"] = NotDiamondSelector()
        if isinstance(args[0], BaseSelector) and not isinstance(
            args[0], NotDiamondSelector
        ):
            LOGGER.warning(warn_msg)
            args[0] = NotDiamondSelector()

        super().__init__(*args, **kwargs)

        # Try to parse MetadataTypes as notdiamond.LLMConfig before submitting queries
        # Will raise ValueError if metadata is not a valid LLMConfig
        for metadata in self._metadatas:
            _ = _tool_metadata_to_llm_config(metadata)
