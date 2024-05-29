# client/llm_connection.py

import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from ollama import Client
from config.ai_config import AIConfig

class LLMConnection:
    def __init__(self):
        """
        Initialize the LLMConnection class with AI configuration.
        """
        self.ai_config = AIConfig()

    @staticmethod
    def connect_ollama():
        """
        Connect to the Ollama LLM with the configured model name.

        Returns:
            Ollama: An instance of the Ollama model.
        """
        ai_model = Ollama(model=AIConfig.model_name())
        return ai_model

    @staticmethod
    def connect_chat_ollama():
        """
        Connect to the ChatOllama model with the configured model name and temperature.

        Returns:
            ChatOllama: An instance of the ChatOllama model.
        """
        ai_model = ChatOllama(model=AIConfig.model_name(), temperature=0)
        return ai_model

    @staticmethod
    def connect_multimodel_ollama():
        """
        Connect to the multimodel Ollama with the configured multi-model name.

        Returns:
            Ollama: An instance of the Ollama multi-model.
        """
        ai_model = Ollama(model=AIConfig.multi_model_name())
        return ai_model

    @staticmethod
    def ollama_client():
        """
        Create and return an Ollama client with the host and model configuration.

        Returns:
            tuple: A tuple containing the Ollama client instance and model name.
        """
        host, model = AIConfig.ollama_host()
        client = Client(host=host)
        return client, model