# load_image_chain.py
from langchain.chains import TransformChain
from load_image import load_image

load_image_chain = TransformChain(
  input_variables=['image_path'],
  output_variables=['image'],
  transform=load_image
)
