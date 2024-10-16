import time

from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer

from cortecs_py import Cortecs
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

### ---  look at doc size todo remove
tokenizer = AutoTokenizer.from_pretrained('neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8')
for doc in docs:
    print(len(tokenizer.encode(doc.page_content)))
#### ---

# this example demonstrates how to use cortecs for high-throughput applications
with DedicatedLLM(client=cortecs, model_name='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8') as llm:

    chain = prompt | llm

    t0, tokens_processed = time.time(), 0  # measure throughput todo remove

    print("Processing data batch-wise ...")
    summaries = chain.batch([{"text": doc.page_content} for doc in docs])
    for summary in summaries:
        print(summary.content + '-------\n\n\n')

        tokens_processed += summary.response_metadata['token_usage']['total_tokens']  # measure throughput todo remove
    print(f'{tokens_processed} in {time.time() - t0} sec')  # measure throughput todo remove
