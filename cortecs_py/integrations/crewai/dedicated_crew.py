from typing import Any, TypeVar

from crewai import Crew

T = TypeVar("T", bound=type[Any])

class DedicatedCrew(Crew):
    instance_id: str
    client: Any

    def _finish_execution(self, final_string_output: str) -> None:
        super()._finish_execution(final_string_output)
        self.client.stop(self.instance_id)
        self.client.delete(self.instance_id)