import requests
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from cortecs.langchain.client import Cortecs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    model_name = 'neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8'
    cortecs = Cortecs(api_base_url='https://develop.cortecs.ai/api/v1')

    # --- use dynamically provisioned llm ---
    # todo use default instance_type
    instance_id, llm = cortecs.start(model_name='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8',
                                     instance_type='NVIDIA_L4_1')

    # todo use roughly half the context length of the model
    prompt = ChatPromptTemplate.from_template("{text}\n\n Translate to english:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=1000, chunk_overlap=0
    )
    book = requests.get("https://www.gutenberg.org/cache/epub/23532/pg23532.txt").text
    chunks = text_splitter.split_text(book)
    translation_chain = prompt | llm
    translations = [None] * len(chunks)
    for i, translation in translation_chain.batch_as_completed([{'text': chunk} for chunk in chunks]):
        translations[i] = translation.content
        print(translation.content)
    open(f'translated_book.txt', 'w').write(' '.join(translations))

    # --- stop ---
    # cortecs.stop(instance_id)
