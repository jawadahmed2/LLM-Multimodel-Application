from IPython.display import display, Markdown
from langchain_core.prompts import PromptTemplate
from task_execution.crewai_agent import AgentTask

class Prompt_Template:
    def __init__(self):
        pass

    def interview_bot_template(self):
        custom_template = """You are a Machine Learning System Design Interview help  AI Assistant. Given the
        following conversation and a follow up question, Give an appropriate response with the ML context given to you/ '.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """

        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

        return CUSTOM_QUESTION_PROMPT

    def crewai_template(self):
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
            file_path="data_preparation/data/crewai_result/features.txt",
        )

        agent2 = AgentTask(
            role="Business Analyst",
            goal="Extract user stories from each feature provided by the Product Owner and provide detailed requirements to the Python Programming Expert.",
            backstory="""I convert the high-level requirements provided by the Product Owner into low-level, independent user stories for the image-to-video conversion app.
            """,
            task_description="Split the high-level features into low-level, independent user stories",
            expected_output="Create a text file of Low-level, independent user stories as a list",
            context=[agent1.task],
            file_path="data_preparation/data/crewai_result/user_stories.txt",
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
            file_path="data_preparation/data/crewai_result/requirements.txt",
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
            file_path="data_preparation/data/crewai_result/output.py",
        )

        return agent1, agent2, agent3, agent4

    def get_image_info_prompt(self):
        vision_prompt = """
                        Given the image, provide the following information:
                        - A count of how many people are in the image
                        - A list of the main objects present in the image
                        - A description of the image
                        """

        return vision_prompt

    def llm_tunning_template(self):
        from langchain.prompts import PromptTemplate
        # Prompt template
        qa_template = """<s>[INST] You are a helpful assistant.
        Use the following context to answer the question below accurately and concisely:
        {context}
        [/INST] </s>{question}
        """

        # Create a prompt instance
        QA_PROMPT = PromptTemplate.from_template(qa_template)

        return QA_PROMPT

