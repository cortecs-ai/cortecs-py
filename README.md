# ⚙️⚡ pycortecs

A thin library to run dynamic LLM flows using [cortecs.ai](https://cortecs.ai).

## Dynamic provisioning

Dynamic provisioning in serverless computing allows you to build complex workflows without managing infrastructure. The
LLM and underlying compute resources are automatically provisioned for the duration of use, providing efficient,
on-demand access. Once the workflow is complete, the infrastructure is shut down, ensuring you only pay for what you
need. This approach maximizes resource utilization in tasks like batch or cron jobs, minimizing token costs. A typical
workflow include:

1. Dynamically provision an LLM,
2. Set up data processing chains using langchain,
3. Load data and do preprocessing,
4. Execute chains,
5. Shutdown dynamically provisioned infrastructure.

```python
from cortecs.langchain.client import Cortecs

cortecs = Cortecs(api_key='<KEY>', secret='<SECRET>')

## --- use dynamically provisioned llm ---
instance_id, llm = cortecs.start(model_name='<MODEL_NAME>')

chain = llm | ...  # set up complex chains
document = ...  # load data, do preprocessing
chain.run(document)  # execute chains

## ---- shutdown ----
cortecs.stop(instance_id)
```

## Example

### Install

```
pip install pycortecs
```

### Translate a book using langchain

First, set up the in environment variables. Use your credentials from [cortecs.ai](https://cortecs.ai). Optionally,
configure the huggingface tokenizer for this particular example.

```
export CORTECS_CLIENT_ID="<YOUR_ID>"
export CORTECS_CLIENT_SECRET="<YOUR_SECRET>"
export TOKENIZERS_PARALLELISM="false"
```

This example shows how to use [langchain](https://python.langchain.com) to configure a simple translation chain. 
The llm is dynamically provisioned and the chain is executed in paralle. 

```python
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from cortecs.langchain.client import Cortecs

# define the model you want to provision
model_name = 'neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8'

if __name__ == '__main__':
    client = Cortecs()

    # --- use dynamically provisioned llm ---
    instance_id, llm = client.start(model_name=model_name, instance_type='NVIDIA_L4_1')

    # set up a simple translation chain
    prompt = ChatPromptTemplate.from_template("{text}\n\n Translate to english:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=1000, chunk_overlap=0
    )
    translation_chain = prompt | llm

    # load a book to translate to english
    book = requests.get("https://www.gutenberg.org/cache/epub/23532/pg23532.txt").text
    chunks = text_splitter.split_text(book)

    # execute chain (in batches) and persist the result
    translations = [None] * len(chunks)
    for i, translation in translation_chain.batch_as_completed([{'text': chunk} for chunk in chunks]):
        translations[i] = translation.content
        print(translation.content)
    open(f'translated_book.txt', 'w').write(' '.join(translations))

    # --- stop ---
    client.stop(instance_id)
```

This simple example showcases the power of dynamic provisioning. We translated X input tokens to Y output tokens in Z minutes.
The llm can be fully utilized in those Z minutes enabling better cost efficiency. Comparing the execution with cloud-APIs from
OpenAI and Meta we see the costs advantage.

#### TODO insert bar chart