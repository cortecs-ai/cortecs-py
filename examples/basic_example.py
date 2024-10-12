from langchain_openai import ChatOpenAI
from cortecs.client import Cortecs

cortecs = Cortecs(api_base_url='https://develop.cortecs.ai/api/v1')
instance_id, llm_info = cortecs.start_and_poll('neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8')
llm = ChatOpenAI(**llm_info)

joke = llm.invoke('Write an essay about dynamic provisioning')
print(joke.content)

cortecs.stop(instance_id)

