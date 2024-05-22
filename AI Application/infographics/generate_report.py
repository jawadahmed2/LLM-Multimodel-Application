from IPython.display import HTML, display


class Generate_Report:
    def __init__(self):
        pass

    def plt_img_base64(self, img_base64):
        """
        Display base64 encoded string as image

        :param img_base64:  Base64 string
        """
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))