from IPython.display import display, Markdown
from langchain_core.prompts import PromptTemplate

class Prompt_Template:
    """
    Class containing various prompt templates.
    """

    def __init__(self):
        pass

    def interview_bot_template(self):
        """
        Template for interview bot.
        """
        custom_template = """You are a Machine Learning System Design Interview help  AI Assistant. Given the
        following conversation and a follow up question, Give an appropriate response with the ML context given to you/ '.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """

        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

        return CUSTOM_QUESTION_PROMPT


    def get_image_info_prompt(self, user_prompt):
        """
        Prompt for image information.
        """
        vision_prompt = f"""
                        Given the image, provide the following information:
                        {user_prompt}
                        """

        return vision_prompt

    def llm_tunning_template(self):
        """
        Template for LLM tuning.
        """
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
        """
        Prompt for graph generation.
        """
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
        """
        Prompt for RAG grading.
        """
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
        """
        Prompt for question rewriting.
        """
        prompt = PromptTemplate(
                template="""You are generating questions that is well optimized for
                            retrieval. \n
                Look at the input and try to reason about the underlying semantic
                intent / meaning. \n
                Here is the initial question:
                \n ------- \n
                {question}
                \n ------- \n
                Provide an improved question without any preamble, only respond
                with the updated question: """,
                input_variables=["question"],
            )

        return prompt