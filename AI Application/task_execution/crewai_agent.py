from data_preparation.data_generation import Data_Generation
from client.llm_connection import LLMConnection
from crewai import Agent, Task, Crew, Process
from pathlib import Path
import subprocess
import os

data_generation = Data_Generation()
llm_connection = LLMConnection()

class AgentTask:
    """
    A class representing an agent's task.

    Attributes:
    - file_name (str): The name of the file to write the output to.
    - agent (Agent): The agent assigned to the task.
    - task (Task): The task to be performed by the agent.
    """

    def __init__(
        self,
        role,
        goal,
        backstory,
        task_description,
        expected_output,
        context=[],
        file_path=None,
    ):
        """
        Initialize the AgentTask class.

        Args:
        - role (str): The role of the agent.
        - goal (str): The goal of the agent.
        - backstory (str): The backstory of the agent.
        - task_description (str): The description of the task.
        - expected_output (str): The expected output of the task.
        - context (list): The context of the task.
        - file_path (str): The file path to write the output to.
        """
        self.file_name = file_path
        # Initialize an Agent instance for this task
        self.agent = Agent(
            role=role, goal=goal, backstory=backstory, allow_delegation=False
        )

        # Create a Task instance based on the provided parameters
        if len(context) == 0:
            self.task = Task(
                description=task_description,
                expected_output=expected_output,
                agent=self.agent,
                callback=self.callback,
            )
        else:
            self.task = Task(
                description=task_description,
                expected_output=expected_output,
                agent=self.agent,
                context=context,
                callback=self.callback,
            )

    def callback(self, output):
        """
        Perform callback actions after executing the task.

        Args:
        - output: The output of the task.

        Returns:
        - str: Error message if any.
        """
        if output is not None:
            if self.file_name is None:
                filename = f'../data_preparation/data/crewai_result/{self.agent.role.replace(" ","")}.py'
                data_generation.write_script_to_file(output.raw_output, file_name=filename)
                # Define the Bash command you want to execute
                bash_command = f'python result/{self.agent.role.replace(" ","")}.py'
                # Execute the Bash command
                process = subprocess.Popen(
                    bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                # Capture the output and errors
                op, error = process.communicate()

                # Check if there were any errors
                if error:
                    print(error.decode())
                    self.task.expected_output = f"fix the bug in the generated script: {self.task.output.raw_output} "
                    crew_bug_fixer = Crew(
                        agents=[
                            self.agent,
                        ],
                        tasks=[
                            self.task,
                        ],
                        llm=llm_connection.connect_ollama(),
                        verbose=True,
                        process=Process.sequential,
                    )
                    result = crew_bug_fixer.kickoff()
                    self.callback(result.output)
                    return error.decode()
                else:
                    return ""
            else:
                file_path = self.file_name
                print(data_generation.write_script_to_file(output.raw_output, file_path))
                print(file_path.endswith("requirements.txt"))
                if file_path.endswith("requirements.txt"):
                    print(f"installing requirements.txt in {file_path}")
                    bash_command = f"pip install -r {file_path}"

                    # Execute the Bash command
                    process = subprocess.Popen(
                        bash_command.split(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    # Capture the output and errors
                    op, error = process.communicate()
                    if error:
                        print(error.decode())
                        return error.decode()
                    else:
                        print(f"Script has been successfully executed")
                        return f"Script has been successfully executed"
                else:
                    return f"Script has been successfully written to {self.file_name}"
