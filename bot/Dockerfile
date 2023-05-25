FROM python:3.11.3-alpine3.18

WORKDIR /bot

COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN pip install pylint

COPY ./src ./src
COPY ./.pylintrc ./.pylintrc

RUN pylint --errors-only /bot/src/bot.py

CMD [ "python", "-u", "./src/bot.py" ]
