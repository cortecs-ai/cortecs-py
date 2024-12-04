from crewai import Agent, Crew, Process, Task
from crewai.project import agent, crew, task

from cortecs_py.integrations import DedicatedCrew, DedicatedCrewBase

# Uncomment the following line to use an example of a custom tool
# from example_crew.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@DedicatedCrewBase
class ExampleCrew:
    """ExampleCrew crew"""
    
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
        return DedicatedCrew(
            instance_id=self.instance_id,
            client=self.client,
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )