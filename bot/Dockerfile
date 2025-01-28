FROM python:3.12.8-alpine3.21

WORKDIR /bot

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./src ./src

CMD [ "python", "-u", "./src/bot.py" ]
