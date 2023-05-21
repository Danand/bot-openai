FROM python:3.11.3-alpine3.18

WORKDIR /bot

COPY requirements.txt .

ARG REQUIREMENTS_MD5

RUN pip install -r requirements.txt

COPY bot.py .

CMD [ "python", "-u", "bot.py" ]
