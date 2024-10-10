# ⚙️⚡ cortecs-py

A thin wrapper around [cortecs.ai](https://cortecs.ai) to support dynamic provisioning.

## Dynamic provisioning

Dynamic provisioning allows you to run LLM-workflows on dedicated compute. The
LLM and underlying resources are automatically provisioned for the duration of use, providing maximum cost-efficiency.
Once the workflow is complete, the infrastructure is automatically shut down. 

This library starts and stops your resources. The logic can be implemented using popular frameworks such as [langchain]() 
or [crewAI]().

1. Load (vast amounts of) data
2. **Start your LLM**
3. Execute your (batch) jobs 
4. **Shutdown your LLM**

```python
from cortecs.client import Cortecs
from cortecs.langchain.dedicated_llm import DedicatedLLM

cortecs = Cortecs()
docs = ...  # load documents

with DedicatedLLM(client=cortecs, model_name='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8') as llm:
    chain = ... | llm  # run arbitrary chains
    result = chain.batch([{"text": doc.page_content} for doc in docs])
```

## Getting started

### Install

```
pip install pycortecs
```

### Summarizing documents

First, set up the in environment variables. Use your credentials from [cortecs.ai](https://cortecs.ai). 

```
export CORTECS_CLIENT_ID="<YOUR_ID>"
export CORTECS_CLIENT_SECRET="<YOUR_SECRET>"
```

This example shows how to use [langchain](https://python.langchain.com) to configure a simple translation chain.
The llm is dynamically provisioned and the chain is executed in paralle.

```python
from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate

from cortecs.client import Cortecs
from cortecs.langchain.dedicated_llm import DedicatedLLM

cortecs = Cortecs(api_base_url='https://develop.cortecs.ai/api/v1')
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=20,
    get_ful_documents=True,
    doc_content_chars_max=25000,  # ~6.25k tokens, make sure the models supports that context length
    load_all_available_meta=False
)

prompt = ChatPromptTemplate.from_template("{text}\n\n Explain me like I'm five:")
docs = loader.load()

with DedicatedLLM(client=cortecs, model_name='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8') as llm:
    chain = prompt | llm

    print("Processing data batch-wise ...")
    summaries = chain.batch([{"text": doc.page_content} for doc in docs])
    for summary in summaries:
        print(summary.content + '-------\n\n\n')
```

This simple example showcases the power of dynamic provisioning. We translated X input tokens to Y output tokens in Z
minutes.
The llm can be fully utilized in those Z minutes enabling better cost efficiency. Comparing the execution with
cloud-APIs from
OpenAI and Meta we see the costs advantage.

#### TODO insert bar chart

## Use Cases

* Batch processing
* Low latency -> [How to process reddit in realtime]()
* Multi-agents -> [How to use CrewAI without request limits]()
* High-security 

For more information see our [docs]() or join our [discord]().
