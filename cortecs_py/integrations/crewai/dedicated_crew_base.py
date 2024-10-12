import os
from typing import Any, Type, TypeVar

from crewai.project import CrewBase

from cortecs_py.client import Cortecs
T = TypeVar("T", bound=Type[Any])

def DedicatedCrewBase(cls: T) -> T:
    class WrappedDedicatedClass(CrewBase(cls)):  # Inherit behavior from CrewBase
        def __init__(self, cortecs_id=None, cortecs_secret=None, instance_params={}, *args, **kwargs):
            self.client = Cortecs(cortecs_id, cortecs_secret)

            if 'model_name' not in instance_params:
                instance_params['model_name'] = os.environ.get('OPENAI_MODEL_NAME').split('openai/')[-1]

            self.instance_id, llm_info = self.client.start_and_poll(**instance_params)
            # crewai uses openai environment variables
            # as cortecs is openai compatible, we just need to set the environment variables accordingly
            os.environ['OPENAI_MODEL_NAME'] = 'openai/' + llm_info['model_name']
            os.environ['OPENAI_API_BASE'] = llm_info['base_url']
            super().__init__(*args, **kwargs)

    return WrappedDedicatedClass