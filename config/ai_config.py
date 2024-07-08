# config/ai_config.py

import os
from dotenv import load_dotenv

class AIConfig:
    def __init__(self):
        """
        Initialize the AIConfig class by loading environment variables from a .env file.
        """
        load_dotenv()
        # Set environment variables for various API keys
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

    @staticmethod
    def model_name():
        """
        Get the default model name for single-model operations.

        Returns:
            str: The name of the default model.
        """
        model_name = 'mistral'
        return model_name

    @staticmethod
    def multi_model_name():
        """
        Get the model name for multi-model operations.

        Returns:
            str: The name of the multi-model.
        """
        model_name = 'bakllava'
        return model_name

    @staticmethod
    def ollama_host():
        """
        Get the host and model configuration for Ollama.

        Returns:
            tuple: A tuple containing the host URL and the model name.
        """
        host = 'http://localhost:11434'
        model = 'mistral:latest'
        return host, model

    @staticmethod
    def embeddings_model():
        """
        Get the embeddings model name.

        Returns:
            str: The name of the embeddings model.
        """
        model = 'nomic-embed-text'
        return model
