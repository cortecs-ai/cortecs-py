import os
from typing import Dict
from langchain_openai import ChatOpenAI

from cortecs_py import Cortecs

class DedicatedLLM:
    def __init__(self, 
        client: Cortecs,
        model_name: str,
        instance_type: str = None,
        context_length: int = None,
        force: bool = False,
        poll_interval: int = 5,
        max_retries: int = 150,
        api_key: str | None = None,
        openai_api_kwargs: Dict | None = None
        ):
        self.client = client
        self.provision_kwargs = {
            "model_name": model_name,
            "instance_type": instance_type,
            "context_length": context_length,
            "force": force,
            "poll_interval": poll_interval,
            "max_retries": max_retries,
        }

        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set `OPENAI_API_KEY` as environment variable or pass it as an argument to DedicatedLLM."
            )

        self.instance_id = None
        self.openai_api_kwargs = openai_api_kwargs or {}

    def __enter__(self) -> ChatOpenAI:
        self.instance_id, llm_info = self.client.start_and_poll(**self.provision_kwargs)
        return ChatOpenAI(
            api_key=self.api_key,
            model=llm_info.get("model_name"),
            base_url=llm_info.get("base_url"),
            **self.openai_api_kwargs,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        # self.client.stop(self.instance_id)  # todo shutdown automatically as soon as instant-provisioning is enabled
        pass
