import os

from langchain_openai import ChatOpenAI

from cortecs_py import Cortecs


class DedicatedLLM:
    def __init__(
        self,
        client: Cortecs,
        model_id: str,
        hardware_type_id: str = None,
        context_length: int = None,
        billing_interval: str = "per_minute",
        poll_interval: int = 5,
        max_retries: int = 150,
        api_key: str | None = None,
        **kwargs: dict[str, any],
    ) -> None:
        self.client = client
        self.provision_kwargs = {
            "model_id": model_id,
            "hardware_type_id": hardware_type_id,
            "context_length": context_length,
            "billing_interval": billing_interval,
            "poll": True,
            "poll_interval": poll_interval,
            "max_retries": max_retries,
        }

        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set `OPENAI_API_KEY` as environment variable or pass it as an argument to DedicatedLLM.")

        self.instance_id = None
        self.openai_api_kwargs = kwargs or {}

    def __enter__(self) -> ChatOpenAI:
        self.instance = self.client.start(**self.provision_kwargs)
        config = self.instance.chat_openai_config(api_key=self.api_key, **self.openai_api_kwargs)
        return ChatOpenAI(**config)

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: type | None) -> bool | None:
        self.client.stop(self.instance.instance_id)
        self.client.delete(self.instance.instance_id)
