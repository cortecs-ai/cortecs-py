import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import trange
from transformers import AutoTokenizer

from cortecs_py import Cortecs
from cortecs_py.integrations import DedicatedLLM

if __name__ == '__main__':
    model_name = 'neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
    cortecs = Cortecs()

    prompt = ChatPromptTemplate.from_template("{text}\n\n Translate to German. Don't comment:")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,
                                                                              chunk_size=200, chunk_overlap=0)
    print('Loading and preprocessing data ...')
    bible = requests.get("https://openbible.com/textfiles/kjv.txt").text
    chunks = text_splitter.split_text(bible)

    with DedicatedLLM(cortecs, model_name, 2000, temperature=0.) as llm:

        translation_chain = prompt | llm
        batch_size = 100  # use batched inference
        for i in trange(0, len(chunks), batch_size, desc='Translating bible ...'):
            batch_of_chunks = [{'text': doc} for doc in chunks[i:i + batch_size]]
            translations = translation_chain.batch(batch_of_chunks)
            translation = ' '.join([t.content for t in translations])
            open(f'translated_bible.txt', 'a').write(translation)
