FROM python:3.11.3-alpine3.18

WORKDIR /bot

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./src ./src

CMD [ "python", "-u", "./src/bot.py" ]
