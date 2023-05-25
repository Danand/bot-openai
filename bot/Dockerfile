FROM python:3.11.3-alpine3.18

WORKDIR /bot

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pylint

COPY ./.pylintrc ./.pylintrc
COPY ./src ./src

RUN pylint --errors-only /bot/src/bot.py

CMD [ "python", "-u", "./src/bot.py" ]
