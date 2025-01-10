import os

from langchain_openai import ChatOpenAI

from cortecs_py import Cortecs


class DedicatedLLM:
    def __init__(
            self,
            client: Cortecs,
            model_name: str,
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
            "model_name": model_name,
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

        self.instance = None
        self.openai_api_kwargs = kwargs or {}

    def start_up(self) -> ChatOpenAI:
        self.instance = self.client.start(**self.provision_kwargs)
        return ChatOpenAI(api_key=self.api_key, model_name=self.instance.instance_args.hf_name,
                          base_url=self.instance.base_url, **self.openai_api_kwargs)

    def shut_down(self) -> None:
        self.client.stop(self.instance.instance_id)
        self.client.delete(self.instance.instance_id)

    def __enter__(self) -> ChatOpenAI:
        return self.start_up()

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: type | None) -> bool | None:
        return self.shut_down()
