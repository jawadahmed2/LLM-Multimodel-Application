from client.llm_connection import LLMConnection
from data_preparation.data_generation import Data_Generation
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent, Tool

llm_connection = LLMConnection()
data_generation = Data_Generation()

class Task_Execution:
    def __init__(self):
        self.ollama = llm_connection.connect_ollama()
        self.chat_ollama = llm_connection.connect_chat_ollama()

    def execute_automate_browsing(self, search_query):
        # Pull the ReAct prompting approach prompt to be used as base
        prompt = data_generation.get_react_prompting()
        serper_wrapper = GoogleSerperAPIWrapper()
        tools = [
            Tool(
                name="Intermediate Answer",
                description="Search Google and return the first result.",
                func=serper_wrapper.run,
            )
        ]
        agent = create_react_agent(self.chat_ollama, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        output = agent_executor.invoke({"input": search_query})

        return output

