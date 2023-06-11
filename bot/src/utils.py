import re

def strip_markdown(text: str) -> str:
    emphasis_regex = re.compile(r'\*\*(.*)\*\*|\*(.*)\*')
    strikethrough_regex = re.compile(r'~~(.*)~~')
    links_and_images_regex = re.compile(r'\[.*?\]\(.*?\)')
    code_regex = re.compile(r'`(.*)`')
    headers_regex = re.compile(r'^#* ')

    text = emphasis_regex.sub(r'\g<1>\g<2>', text)
    text = strikethrough_regex.sub(r'\g<1>', text)
    text = links_and_images_regex.sub('', text)
    text = code_regex.sub(r'\g<1>', text)
    text = headers_regex.sub('', text)

    return text
