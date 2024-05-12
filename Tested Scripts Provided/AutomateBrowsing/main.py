from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the environment for AI model interaction
os.environ["OPENAI_API_KEY"] = "............" # add your open ai key
os.environ["SERPER_API_KEY"] = "732570f9a9b4c477887c6ced8dc7798b1c8e44bb"   #here i am adding my key but you can create your own on google serper
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "mistral"

# Import an AI model for content generation and analysis
from langchain_community.llms import Ollama

# Initialize the AI model with a specific configuration
ai_model = Ollama(model="mistral")

# Pull the ReAct prompting approach prompt to be used as base
prompt = hub.pull("hwchase17/react")

## search.run is a parsed version of search.results function

search = GoogleSerperAPIWrapper()

# Create a Tool object so the LLM can decide when to use which tools
tools = [
    Tool(
        name="Intermediate Answer",
        description="Search Google and return the first result.",
        func=search.run,
    )
]

###### Comment out the following code in order to not used the OpenAI model
# model = 'gpt-3.5-turbo'
# llm = ChatOpenAI(temperature=0, model=model, verbose=False)

# Construct the ReAct agent
agent = create_react_agent(ai_model, tools, prompt) # Use the local Ollama Mistral model

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

search_query = "What is happening with Arvind Kejriwal today?"

output = agent_executor.invoke({"input": search_query})

print(output)
