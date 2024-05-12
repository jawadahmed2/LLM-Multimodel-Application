import os
from dotenv import load_dotenv

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

openai_key = ""
tavily_api_key = "" # I added my own API key here you may generate your own later on

os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["OPENAI_API_KEY"] = openai_key