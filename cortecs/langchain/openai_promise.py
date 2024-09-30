from langchain_openai import ChatOpenAI

from cortecs.client import BaseCortecs


class OpenAIPromise(ChatOpenAI):
    status_client: BaseCortecs
    status_kwargs: dict
    provisioned: bool = False

    def _wait_until_provisioned(self):
        self.provisioned = True
        self.status_client.start_and_poll(**self.status_kwargs)

    def invoke(self, *args, **kwargs):
        if not self.provisioned:
            self._wait_until_provisioned()
        return super(OpenAIPromise, self).invoke(*args, **kwargs)

