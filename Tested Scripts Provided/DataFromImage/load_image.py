#load_image.py
import base64
import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


def load_image(inputs: dict) -> dict:
  "Load image from file and encode it as base64."
  image_path = inputs["image_path"]

  # def encode_image(image_path):
  #   with open(image_path, "rb") as image_file:
  #     encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
  #   return encoded_image
  # image_base64 = encode_image(image_path)
  pil_image = Image.open(image_path)
  image_base64 = convert_to_base64(pil_image)
  # plt_img_base64(image_base64)
  return {"image": image_base64}
