# cortecs-py

Lightweight wrapper for the [cortecs.ai](https://cortecs.ai) enabling instant provisioning.

## ⚡Quickstart
Dynamic provisioning allows you to run LLM-workflows on dedicated compute. The
LLM and underlying resources are automatically provisioned for the duration of use, providing maximum cost-efficiency.
Once the workflow is complete, the infrastructure is automatically shut down. 

This library starts and stops your resources. The logic can be implemented using popular frameworks such as [LangChain](https://python.langchain.com) 
or [crewAI](https://docs.crewai.com/introduction).

1. **Start your LLM**
2. Execute your (batch) jobs
3. **Shutdown your LLM**

```python
from cortecs_py.client import Cortecs
from cortecs_py.integrations import DedicatedLLM

cortecs = Cortecs()

with DedicatedLLM(client=cortecs, model_name='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8') as llm:
    essay = llm.invoke('Write an essay about dynamic provisioning')
    print(essay.content)

```

## Example

### Install

```
pip install cortecs-py
```

### Summarizing documents

First, set up the environment variables. Use your credentials from [cortecs.ai](https://cortecs.ai). 
```
export OPENAI_API_KEY="<YOUR_CORTECS_API_KEY>"
export CORTECS_CLIENT_ID="<YOUR_ID>"
export CORTECS_CLIENT_SECRET="<YOUR_SECRET>"
```
This example shows how to use [LangChain](https://python.langchain.com) to configure a simple translation chain.
The llm is dynamically provisioned and the chain is executed in parallel.

```python
from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate

from cortecs_py.client import Cortecs
from cortecs_py.integrations import DedicatedLLM

cortecs = Cortecs()
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=20,
    get_ful_documents=True,
    doc_content_chars_max=25000,  # ~6.25k tokens, make sure the models supports that context length
    load_all_available_meta=False
)

prompt = ChatPromptTemplate.from_template("{text}\n\n Explain to me like I'm five:")
docs = loader.load()

with DedicatedLLM(client=cortecs, model_name='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8') as llm:
    chain = prompt | llm

    print("Processing data batch-wise ...")
    summaries = chain.batch([{"text": doc.page_content} for doc in docs])
    for summary in summaries:
        print(summary.content + '-------\n\n\n')
```

This simple example showcases the power of dynamic provisioning. We summarized **128.6k input tokens** into **7.9k output tokens** in **35
seconds**.
The llm can be **fully utilized** in those 35 seconds enabling better cost efficiency.


## Use Cases

* Batch processing
* Low latency -> [How to process reddit in realtime]()
* Multi-agents -> [How to use CrewAI without request limits]()
* High-security 

For more information see our [docs]() or join our [discord]().
