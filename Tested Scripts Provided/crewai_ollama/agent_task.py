from crewai import Agent, Task, Crew, Process
from pathlib import Path
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

# Set up the environment for AI model interaction
os.environ["OPENAI_API_KEY"] = ".........................." # Add your OpenAI API key here
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "mistral"

# Import an AI model for content generation and analysis
from langchain_community.llms import Ollama

# Initialize the AI model with a specific configuration
ai_model = Ollama(model="mistral")


def write_script_to_file(script_content: str, file_name: str) -> str:
    """
    Write the script content to a file.
    """
    # Extract script content between ```python and ```
    start_index = script_content.find("```python")
    l1 = len("```python")
    if start_index == -1:
        start_index = script_content.find("Action:")
        l1 = len("Action:")
    if start_index == -1:
        start_index = script_content.find("```bash")
        l1 = len("```bash")
    if start_index == -1:
        start_index = script_content.find("```markdown")
        l1 = len("```markdown")
    if start_index == -1:
        start_index = script_content.find("```makefile")
        l1 = len("```makefile")
    if start_index == -1:
        start_index = script_content.find("```text")
        l1 = len("```text")
    if start_index == -1:
        start_index = script_content.find("```")
        l1 = len("```")
    end_index = script_content.find("```", start_index + l1)
    if end_index == -1:
        end_index = len(script_content) - 1
    if start_index == -1 or end_index == -1:
        return "Script content must be enclosed between ```python and ``` tags."
    else:
        script_content = script_content[start_index + l1 : end_index].strip()

        # Write the extracted script content to the file
        file_path = Path(file_name)
        with open(file_path, "w") as file:
            file.write(script_content)
        return script_content


class AgentTask:

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
        if output is not None:
            if self.file_name is None:
                filename = f'result/{self.agent.role.replace(" ","")}.py'
                write_script_to_file(output.raw_output, file_name=filename)
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
                        llm=ai_model,
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
                print(write_script_to_file(output.raw_output, file_path))
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
