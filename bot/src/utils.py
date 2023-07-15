import re

def strip_markdown(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("~~", "")
    text = text.replace("`", "")
    text = text.replace("`", "")

    return text

def split_string_to_batches(string, max_length):
    return [string[i: i + max_length] for i in range(0, len(string), max_length)]
