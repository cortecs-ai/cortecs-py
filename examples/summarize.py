from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate

from cortecs_py import Cortecs
from cortecs_py.integrations import DedicatedLLM

cortecs = Cortecs()
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=20,
    get_ful_documents=True,
    doc_content_chars_max=25000,  # ~6.25k tokens, make sure the models supports that context length
    load_all_available_meta=False,
)

prompt = ChatPromptTemplate.from_template("{text}\n\n Explain to me like I'm five:")
docs = loader.load()

with DedicatedLLM(client=cortecs, model_id="neuralmagic--Meta-Llama-3.1-8B-Instruct-FP8") as llm:
    chain = prompt | llm

    print("Processing data batch-wise ...")
    summaries = chain.batch([{"text": doc.page_content} for doc in docs])
    for summary in summaries:
        print(summary.content + "-------\n\n\n")
