from IPython.display import display, Markdown
from langchain import PromptTemplate


class Generate_Report:
    def __init__(self):
        pass

    def interview_bot_report(self):

        # Generate report
        print("Report generated")

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
