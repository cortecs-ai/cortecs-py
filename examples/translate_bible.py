import os

import requests
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import trange
from transformers import AutoTokenizer
from cortecs.client import BaseCortecs


client_id = os.environ.get('CORTECS_CLIENT_ID')
client_secret = os.environ.get('CORTECS_CLIENT_SECRET')
api_key = os.environ.get('CORTECS_API_KEY')


if __name__ == '__main__':

    model_name = 'neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8'
    bible = requests.get("https://openbible.com/textfiles/kjv.txt").text

    # split bible into chunks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=8000, chunk_overlap=0
    )
    bile_docs = text_splitter.split_text(bible)

    # start a dedicated instance
    client = BaseCortecs(api_base_url='https://develop.cortecs.ai/api/v1')
    instance = client.start_and_poll(model_name=model_name, instance_type='NVIDIA_L40s_1')

    # create processing chain
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
    client.stop(instance['id'])
