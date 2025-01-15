import os
from typing import Any
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.project import agent, crew, task, after_kickoff, CrewBase

from cortecs_py import Cortecs

# Uncomment the following line to use an example of a custom tool
# from example_crew.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

load_dotenv()

@CrewBase
class ExampleCrew:
    
    def __init__(self) -> None:
        self.start_llm()
    
    def start_llm(self) -> None:
        self.cortecs_client = Cortecs()
        self.model = os.environ["MODEL"].removeprefix("openai/")
        
        print(f"Starting model {self.model}...")
        self.instance = self.cortecs_client.ensure_instance(self.model)
        os.environ["OPENAI_API_BASE"] = self.instance.base_url
    
    @after_kickoff
    def stop_and_delete_llm(self, result: Any) -> Any:  # noqa: ANN401
        self.cortecs_client.stop(self.instance.instance_id)
        self.cortecs_client.delete(self.instance.instance_id)
        print(f"Model {self.model} stopped and deleted.")

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            # tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(config=self.agents_config["reporting_analyst"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        """Creates the ExampleCrew crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
