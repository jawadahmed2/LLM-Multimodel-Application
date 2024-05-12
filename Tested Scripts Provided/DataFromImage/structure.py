#structure.py
from langchain_core.pydantic_v1 import BaseModel, Field

class ImageInformation(BaseModel):
  "Information about an image."
  image_description: str = Field(description="a short description of the image")
  people_count: int = Field(description="number of humans on the picture")
  main_objects: list[str] = Field(description="list of the main objects on the picture")
