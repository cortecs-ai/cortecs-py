# ⚙️⚡ pycortecs

Thin wrapper to manage [cortecs.ai](https://cortecs.ai) LLM workflows.

## LLMs for Big Data

Dedicated inference allows you to process large volumes of unstructured data efficiently:

- **Quantization**: By reducing precision to FP8, you cut compute costs by 50% while retaining 99.9% of the original
  model quality.
- **Optimized batch processing**: Instances are fine-tuned for maximum performance when handling batch workloads.
- **Latest generation**: Leverage the latest GPUs (L4, L40s, H100) to accelerate throughput on demanding tasks.

## Example

This example shows how to translate the entire Bible into German:
1. Download the bible,
2. split bible into chunks,
3. start an LLM inference endpoint,
4. Execute the translation chain,
5. finally, stop the inference endpoint.

```python
import os

import requests
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import trange
from transformers import AutoTokenizer
from cortecs.client import Client


client_id = os.environ.get('CORTECS_CLIENT_ID')
client_secret = os.environ.get('CORTECS_CLIENT_SECRET')
api_key = os.environ.get('CORTECS_API_KEY')


if __name__ == '__main__':

    model_name = 'neuralmagic/Mistral-Nemo-Instruct-2407-FP8'
    bible = requests.get("https://openbible.com/textfiles/kjv.txt").text

    # split bible into chunks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=8000, chunk_overlap=0
    )
    bile_docs = text_splitter.split_text(bible)

    # start a dedicated instance
    client = Client()
    instance = client.start_instance_and_poll(model_name=model_name, instance_type='NVIDIA_H100_1')

    # create chain
    llm = OpenAI(openai_api_key=api_key,
                 openai_api_base=f'https://{instance["domain"]}/v1',
                 model_name=model_name)
    prompt = ChatPromptTemplate.from_template("<s>[INST] Translate the following excerpt of the bible to {language}.\n{text}[/INST]")
    translation_chain = prompt | llm

    batch_size = 10  # use batched inference
    for i in trange(0, len(bile_docs), batch_size):
        batch_of_docs = [{'text': doc, 'language': 'German'} for doc in bile_docs[i:i + batch_size]]
        translated_docs = translation_chain.batch(batch_of_docs)

        with open(f'bible_German.txt', 'a') as file:
            translation = ' '.join(translated_docs)
            file.write(translation)

    # shutdown instance
    client.stop_instance(instance['id'])
```