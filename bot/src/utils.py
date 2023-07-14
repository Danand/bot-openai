import re

def strip_markdown(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("~~", "")
    text = text.replace("`", "")
    text = text.replace("`", "")

    return text
