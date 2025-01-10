from openai import OpenAI
from cortecs_py import Cortecs

cortecs = Cortecs()
my_model = 'cortecs/phi-4-FP8-Dynamic'

# --> Start an instance
my_instance = cortecs.start(my_model)
client = OpenAI(base_url=my_instance.base_url)

completion = client.chat.completions.create(
  model=my_model,
  messages=[
    {"role": "user", "content": "Write a joke about LLMs."}
  ]
)
print(completion.choices[0].message.content)

# --> Stop the instance
cortecs.stop(my_instance.instance_id)