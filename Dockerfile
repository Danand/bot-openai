FROM python:3.10.6-alpine3.16

WORKDIR /bot

COPY requirements.txt .

ARG REQUIREMENTS_MD5

RUN pip install -r requirements.txt

COPY bot.py .

CMD [ "python", "-u", "bot.py" ]
