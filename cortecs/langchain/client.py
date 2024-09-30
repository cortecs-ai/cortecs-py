import os

from langchain_openai import ChatOpenAI

from cortecs.client import BaseCortecs
from cortecs.langchain.openai_promise import OpenAIPromise


class Cortecs:
    def __init__(self, api_key: str = None, *args, **kwargs):
        self.__api_key = api_key if api_key else os.environ.get('CORTECS_API_KEY')
        self._client = BaseCortecs(*args, **kwargs)

    def start_and_poll(self, *args, **kwargs):
        instance_status = self._client.start_and_poll(*args, **kwargs)
        return instance_status['id'], ChatOpenAI(openai_api_key=self.__api_key,
                                             openai_api_base=f'https://{instance_status["domain"]}/v1',
                                             model_name=instance_status['model_id'].replace('--', '/'))

    def start(self, openai_kwargs={}, **kwargs):
        instance_status = self._client.start(**kwargs)
        return instance_status['id'], OpenAIPromise(status_client=self._client,
                                                    openai_api_key=self.__api_key,
                                                    openai_api_base=f'https://{instance_status["domain"]}/v1',
                                                    model_name=instance_status['model_id'].replace('--', '/'),
                                                    status_kwargs=kwargs,
                                                    **openai_kwargs)

    def stop(self, *args, **kwargs):
        return self._client.stop(*args, **kwargs)
