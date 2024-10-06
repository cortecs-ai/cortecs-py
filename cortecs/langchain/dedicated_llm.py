import os

from langchain_openai import ChatOpenAI


class DedicatedLLM:
    def __init__(self, client, model_name, instance_type=None, context_length=None, api_key=None, **kwargs):
        self.client = client
        self.model_name = model_name
        self.instance_type = instance_type
        self.context_length = context_length
        self.api_key = api_key if api_key else os.environ.get('CORTECS_API_KEY')
        self.instance_id = None
        self.open_api_kwargs = kwargs

    def provision(self):
        instance_status = self.client.start_and_poll(self.model_name, self.instance_type, self.context_length)
        openai_api_base = f'https://{instance_status["domain"]}/v1'
        return instance_status['id'], ChatOpenAI(openai_api_key=self.api_key,
                          openai_api_base=openai_api_base,
                          model_name=self.model_name, **self.open_api_kwargs)

    def shutdown(self, instance_id):
        self.client.stop(instance_id)

    def __enter__(self):
        self.instance_id, llm = self.provision()
        return llm

    def __exit__(self, exc_type, exc_value, traceback):
        # self.client.stop(self.instance_id)
        pass