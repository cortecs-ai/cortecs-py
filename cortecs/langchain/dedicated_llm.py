import os

from langchain_openai import ChatOpenAI


class DedicatedLLM:
    def __init__(self, client, request_conf={}, api_key=None, **kwargs):
        self.client = client
        self.provision_kwargs = kwargs
        self.api_key = api_key if api_key else os.environ.get('OPENAI_API_KEY')
        self.instance_id = None
        self.open_api_kwargs = request_conf  # e.g. temperature


    def __enter__(self):
        self.instance_id, llm_info = self.client.start_and_poll(**self.provision_kwargs)
        return ChatOpenAI(api_key=self.api_key, **llm_info, **self.open_api_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        # self.client.stop(self.instance_id)  # todo shutdown automatically as soon as instant-provisioning is enabled
        pass
