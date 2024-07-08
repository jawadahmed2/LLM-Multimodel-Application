#image_model.py
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain
from langchain import globals
from structure import ImageInformation
import json
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from load_image import load_image
# Load environment variables from .env file
load_dotenv()

# Set up the environment for AI model interaction
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "bakllava"


# Initialize the AI model with a specific configuration
ai_model = Ollama(model="bakllava")

parser = JsonOutputParser(pydantic_object=ImageInformation)


@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    image = load_image(inputs)["image"]
    ai_model.bind(images=image)
    msg = ai_model.invoke(
              [
                    {"role": "system", "content": "you are a usefull assistant that provides information about images"},
                    {"role": "user", "content": inputs["prompt"]},
                ]

    )
    return msg


