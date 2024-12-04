from langchain_openai import ChatOpenAI

from cortecs_py.client import Cortecs

cortecs = Cortecs()
instance = cortecs.start("neuralmagic--Meta-Llama-3.1-8B-Instruct-FP8", poll=True)
llm = ChatOpenAI(**instance.chat_openai_config())

joke = llm.invoke("Write a joke about LLMs.")
print(joke.content)

cortecs.stop(instance.instance_id)
