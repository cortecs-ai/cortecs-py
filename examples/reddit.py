import praw
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cortecs.client import Cortecs
from cortecs.langchain.dedicated_llm import DedicatedLLM

# this example demonstrates dedicated inference in realtime settings
if __name__ == '__main__':
    model_name = 'neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
    cortecs = Cortecs(api_base_url='https://develop.cortecs.ai/api/v1')
    reddit = praw.Reddit(user_agent='Read-only example bot')

    with DedicatedLLM(cortecs, model_name, context_length=1500, temperature=0.) as llm:  # todo decrease context_length

        prompt = ChatPromptTemplate.from_template("""
        Given the reddit post below, classify it as either `Art`, `Finance`, `Science`, `Taylor Swift` or `Other`.
        Do not provide an explanation.
        
        {channel}: {title}\n Classification:""")
        classification_chain = prompt | llm | StrOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the biggest Taylor Swift fan."),
            ("user", "Respond to this post:\n {comment}")
        ])
        response_chain = prompt | llm

        # scan reddit in realtime and shill about tay tay
        for post in reddit.subreddit("all").stream.comments():
            topic = classification_chain.invoke({'channel': post.subreddit_name_prefixed, 'title': post.link_title})
            print(f'{post.subreddit_name_prefixed} {post.link_title}')
            if topic == 'Taylor Swift':
                response = response_chain.invoke({'comment': post.body})
                print(post.body + '\n---> ' + response.content)


