import base64

def format_images_to_openai_content(image_path, mime_type='image/jpeg'):
    """
    Formats an image to be used as input for OpenAI's ChatGPT API.
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_chat_gpt_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }

        return image_chat_gpt_content