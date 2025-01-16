from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class EfaxAssistant:
    """EfaxAssistant crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def document_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config["document_classifier"],
            verbose=True,
        )

    @agent
    def medical_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["medical_analyzer"],
            verbose=True,
        )

    @agent
    def alignment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["alignment_agent"],
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def document_classification_task(self) -> Task:
        return Task(
            config=self.tasks_config["document_classification_task"],
        )

    @task
    def medical_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["medical_analysis_task"], output_file="report.md"
        )

    @task
    def alignment_task(self) -> Task:
        return Task(config=self.tasks_config["alignment_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        """Creates the EfaxAssistant crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
