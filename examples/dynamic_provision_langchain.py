from cortecs_py.client import Cortecs
from cortecs_py.integrations.langchain import DedicatedLLM

cortecs = Cortecs()

with DedicatedLLM(cortecs, 'cortecs/phi-4-FP8-Dynamic') as llm:
    joke = llm.invoke('Write a joke about LLMs.')
    print(joke.content)

