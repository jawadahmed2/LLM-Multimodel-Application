from crewai import Process, Crew
from agent_task import AgentTask, ai_model

project_brief = """
Project Title: Image to video conversion app
app type: console
language: python
description: Convert 10 images in a local folder to a video. The video is saved to the same or a different local folder.
"""

# Define Agents and tasks
agent1 = AgentTask(
    role="Product Owner",
    goal=f"Provide features to the Business Analyst for an image-to-video console app. Project briefing is as follows:  {project_brief}. Understand only features needed for the project here.",
    backstory="""I provide high-level requirements to the Business Analyst for the image-to-video console app.""",
    task_description="Extract high-level requirement as features from the Product Owner",
    expected_output="create a text file proper layout with Feature list of high-level requirements",
    file_path="result/features.txt",
)

agent2 = AgentTask(
    role="Business Analyst",
    goal="Extract user stories from each feature provided by the Product Owner and provide detailed requirements to the Python Programming Expert.",
    backstory="""I convert the high-level requirements provided by the Product Owner into low-level, independent user stories for the image-to-video conversion app.
    """,
    task_description="Split the high-level features into low-level, independent user stories",
    expected_output="Create a text file of Low-level, independent user stories as a list",
    context=[agent1.task],
    file_path="result/user_stories.txt",
)
agent3 = AgentTask(
    role="Team Lead",
    goal="Extract dependencies from userstories and create dependency installation scripts",
    backstory="""I create dependency installation commands as a requirements.txt file
    """,
    task_description="create requirements.txt containing dependent libraries from user stories",
    expected_output="""create requirement.txt for dependencies. Make sure to include all the dependencies from the user stories.
     There should not be any conflicting dependencies.Make sure that dependencies are compatible with the python version and is present in repositories.
     Make sure to include the version of the dependencies and "No matching distribution found" error for the dependencies should not be there.
    """,
    context=[agent2.task],
    file_path="result/requirements.txt",
)

agent4 = AgentTask(
    role="Python Programming Expert",
    goal="Develop code in Python based on the Business Analyst",
    backstory="""I Python code for each user story to start 
    implementing the image-to-video conversion app.""",
    task_description="Write python code for each user story to implement the image-to-video console app.",
    expected_output="""complete solution with python code that satisfies each user story and its respective requirements. 
    There should be comments in the code to explain the parts dealing with each user story.
    All user stories must be covered.Make sure the code is bug-free and well-formatted.
    """,
    context=[agent3.task],
    file_path="result/output.py",
)

# Initialize a Crew with Agents, Tasks, and set the process to hierarchical
crew = Crew(
    agents=[agent1.agent, agent2.agent, agent3.agent, agent4.agent],
    tasks=[agent1.task, agent2.task, agent3.task, agent4.task],
    llm=ai_model,
    process=Process.sequential,
    verbose=True,
)

# Execute the crew's process and obtain the result
result = crew.kickoff()

# Print the results of a specific task
print(
    f"""
    Task completed!
    Task: {agent4.task.description}
    Output: {result}
"""
)
