#get_image_informations.py
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from structure import ImageInformation
from load_image_chain import load_image_chain
from image_model import image_model



def get_image_informations(image_path: str) -> dict:
  vision_prompt = """
  Given the image, provide the following information:
  - A count of how many people are in the image
  - A list of the main objects present in the image
  - A description of the image
  """

  vision_chain = load_image_chain | image_model
  output = vision_chain.invoke({'image_path': f'{image_path}', 'prompt': vision_prompt})
  print('Input Prompt:', vision_prompt)
  return output
