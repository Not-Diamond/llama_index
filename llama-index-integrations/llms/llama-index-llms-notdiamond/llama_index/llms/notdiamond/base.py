import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from notdiamond import NotDiamond, LLMConfig
from notdiamond.exceptions import CreateUnavailableError


class NotDiamond(OpenAILike):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://not-diamond-server.onrender.com",
        is_chat_model: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Python client for NotDiamond only supports chat model interactions with `pip install notdiamond[create]`.
        """
        api_key = api_key or os.getenv("NOTDIAMOND_API_KEY")
        self._ndclient = NotDiamond(api_key=api_key)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            **kwargs
        )

    async def amodel_select(self, *args, **kwargs):
        return self._ndclient.amodel_select(*args, **kwargs)

    def model_select(self, *args, **kwargs):
        return self._ndclient.model_select(*args, **kwargs)

    def validate_params(self, *args, **kwargs):
        return self._ndclient.validate_params(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return self._ndclient.bind_tools(*args, **kwargs)

    def call_callbacks(self, *args, **kwargs):
        return self._ndclient.call_callbacks(*args, **kwargs)

    def create(self, *args, **kwargs):
        return self._ndclient.create(*args, **kwargs)

    async def acreate(self, *args, **kwargs):
        return self._ndclient.acreate(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._ndclient.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return self._ndclient.ainvoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._ndclient.stream(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        return self._ndclient.astream(*args, **kwargs)

    @property
    def default_llm(self) -> LLMConfig:
        return self._ndclient.default_llm

    @classmethod
    def class_name(cls) -> str:
        return "NotDiamondLLM"