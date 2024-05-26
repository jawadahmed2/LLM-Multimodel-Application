# config/ai_config.py
import os
from dotenv import load_dotenv

class AIConfig:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

    @staticmethod
    def model_name():
        model_name = 'mistral'
        return model_name

    @staticmethod
    def multi_model_name():
        model_name = 'bakllava'
        return model_name

    @staticmethod
    def ollama_host():
        host = 'http://localhost:11434'
        return host