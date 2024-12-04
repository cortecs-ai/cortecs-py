import os
from typing import TypeVar

from crewai.project import CrewBase

from cortecs_py.client import Cortecs
from cortecs_py.utils import convert_model_name

T = TypeVar("T", bound=type[any])


def DedicatedCrewBase(cls: T) -> T:  # noqa: N802
    class WrappedDedicatedClass(CrewBase(cls)):  # Inherit behavior from CrewBase
        def __init__(self, *args: list[any], **kwargs: dict[str, any]) -> None:
            self.client = Cortecs()

            model_id = os.environ.get('OPENAI_MODEL_NAME', '').split('openai/')[-1]

            instance = self.client.start(model_id=model_id)
            self.instance_id = instance.instance_id
            
            # Set environment variables for OpenAI compatibility
            os.environ['OPENAI_MODEL_NAME'] = 'openai/' + convert_model_name(model_id, to_hf_format=True)
            os.environ['OPENAI_API_BASE'] = instance.base_url

            super().__init__(*args, **kwargs)
        
    return WrappedDedicatedClass