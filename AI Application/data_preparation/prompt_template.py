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

    def graphPrompt(self, input: str, chunk_id):

        # model_info = client.show(model_name=model)
        # print( chalk.blue(model_info))

        SYS_PROMPT = (
            "You are a network graph maker who extracts terms and their relations from a given context. "
            "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
            "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
            "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include person (agent), location, organization, date, duration, \n"
            "\tcondition, concept, object, entity  etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
            "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
            "Thought 3: Find out the relation between each such related pair of terms. \n\n"
            "Format your output as a list of json. Each element of the list contains a pair of terms"
            "and the relation between them like the follwing. NEVER change the value of the chunk_ID as defined in this prompt: \n"
            "[\n"
            "   {\n"
            '       "chunk_id": "CHUNK_ID_GOES_HERE",\n'
            '       "node_1": "A concept from extracted ontology",\n'
            '       "node_2": "A related concept from extracted ontology",\n'
            '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
            "   }, {...}\n"
            "]"
        )
        SYS_PROMPT = SYS_PROMPT.replace("CHUNK_ID_GOES_HERE", chunk_id)

        USER_PROMPT = f"context: ```{input}``` \n\n output: "

        return USER_PROMPT, SYS_PROMPT

    def get_rag_prompt(self, parser):
        prompt = PromptTemplate(
                    template="""You are a grader assessing relevance of a retrieved
                                    document to a user question. \n
                    Here is the retrieved document: \n\n {context} \n\n
                    Here is the user question: {question} \n
                    If the document contains keywords related to the user question,
                        grade it as relevant. \n
                    It does not need to be a stringent test. The goal is to filter out
                    erroneous retrievals. \n
                    Give a binary score 'yes' or 'no' score to indicate whether the
                    document is relevant to the question. \n
                    Provide the binary score as a JSON with no premable or
                    explaination and use these instructons to format the output:
                    {format_instructions}""",
                    input_variables=["query"],
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()
                    },
                )
        return prompt

    def question_rewriter_prompt(self):
        prompt = PromptTemplate(
                template="""You are generating questions that is well optimized for
                            retrieval. \n
                Look at the input and try to reason about the underlying sematic
                intent / meaning. \n
                Here is the initial question:
                \n ------- \n
                {question}
                \n ------- \n
                Provide an improved question without any premable, only respond
                with the updated question: """,
                input_variables=["question"],
            )

        return prompt