from langchain import hub
from client.llm_connection import LLMConnection


llm_connection = LLMConnection()

class Data_Generation:
    def __init__(self):
        self.ollama = llm_connection.connect_ollama()

    # Get the react prompting for the Automate Browsing Task
    @staticmethod
    def get_react_prompting():
        prompt = hub.pull("hwchase17/react")
        return prompt

    def generate_result(self, query):
        result = self.ollama(query)
        return result