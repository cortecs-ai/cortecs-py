from typing import Any, Type, TypeVar
from crewai import Crew

T = TypeVar("T", bound=Type[Any])

class DedicatedCrew(Crew):
    instance_id: str
    client: Any

    def _finish_execution(self, final_string_output):
        super(DedicatedCrew, self)._finish_execution(final_string_output)
        # self.client.stop(self.instance_id)


