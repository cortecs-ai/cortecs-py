# cortecs-py

![PyPI Version](https://img.shields.io/pypi/v/cortecs-py.svg) 
![Python Versions](https://img.shields.io/pypi/pyversions/cortecs-py.svg) 
![Downloads](https://img.shields.io/pypi/dm/cortecs-py.svg) 
![Workflow Status](https://github.com/cortecs-ai/cortecs-py-ci/actions/workflows/test.yaml/badge.svg) 
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://docs.cortecs.ai/) 
[![Join our Discord](https://img.shields.io/badge/discord-join%20chat-7289da.svg)](https://discord.com/invite/bPFEFcWBhp)

Lightweight wrapper for the [cortecs.ai](https://cortecs.ai) enabling instant provisioning.

## ⚡Quickstart
Dynamic provisioning allows you to run LLM-workflows on dedicated compute. The
LLM and underlying resources are automatically provisioned for the duration of use, providing maximum cost-efficiency.
Once the workflow is complete, the infrastructure is automatically shut down. 

This library starts and stops your resources. The logic can be implemented using popular frameworks such as [LangChain](https://python.langchain.com) 
or [crewAI](https://docs.crewai.com/introduction).

1. **Start your LLM**
2. Execute (massive batch) jobs
3. **Shutdown your LLM**

```python
from cortecs_py.client import Cortecs
from cortecs_py.integrations.langchain import DedicatedLLM

cortecs = Cortecs()

with DedicatedLLM(client=cortecs, model_name='cortecs/phi-4-FP8-Dynamic') as llm:
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
This example shows how to use [LangChain](https://python.langchain.com) to configure a simple summarization chain.
The llm is dynamically provisioned and the chain is executed in parallel.

```python
from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate

from cortecs_py.client import Cortecs
from cortecs_py.integrations.langchain import DedicatedLLM

cortecs = Cortecs()
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=40,
    get_ful_documents=True,
    doc_content_chars_max=25000,  # ~6.25k tokens, make sure the models supports that context length
    load_all_available_meta=False
)

prompt = ChatPromptTemplate.from_template("{text}\n\n Explain to me like I'm five:")
docs = loader.load()

with DedicatedLLM(client=cortecs, model_name='cortecs/phi-4-FP8-Dynamic') as llm:
    chain = prompt | llm

    print("Processing data batch-wise ...")
    summaries = chain.batch([{"text": doc.page_content} for doc in docs])
    for summary in summaries:
        print(summary.content + '-------\n\n\n')
```

This simple example showcases the power of dynamic provisioning. We summarized **224.2k input tokens** into **12.9k output tokens** in **55
seconds**.
The llm can be **fully utilized** in those 55 seconds enabling better cost efficiency. Comparing to serverless open source model providers we observe the following:

<img src="https://github.com/user-attachments/assets/3d50d642-9f78-4336-a1a5-235b109d5f68" alt="Price Comparison (USD)" width="400" />
<img src="https://github.com/user-attachments/assets/6dd22261-47ad-40c8-a647-4ee0ab071545" alt="Price Comparison per Million Tokens (USD)" width="400" />

## Use Cases

* Low latency -> [How to process reddit in realtime](https://github.com/cortecs-ai/cortecs-py/blob/main/examples/reddit.py)
* Multi-agents -> [How to use CrewAI without request limits](https://github.com/cortecs-ai/cortecs-py/tree/main/examples/example_crew)
* Batch processing
* High-security 

For more information see our [docs](https://docs.cortecs.ai/) or join our [discord](https://discord.gg/bPFEFcWBhp).
