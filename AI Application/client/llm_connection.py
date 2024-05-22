# client/llm_connection.py
import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

from config.ai_config import AIConfig

class LLMConnection:
    def __init__(self):
        self.ai_config = AIConfig()

    @staticmethod
    def connect_ollama():
        ai_model = Ollama(model=AIConfig.model_name())
        return ai_model

    @staticmethod
    def connect_chat_ollama():
        ai_model = ChatOllama(model=AIConfig.model_name())
        return ai_model

    @staticmethod
    def connect_mulimodel_ollama():
        ai_model = Ollama(model=AIConfig.multi_model_name())
        return ai_model
