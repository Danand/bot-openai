from os import environ as env
from typing import List

import openai

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Filter
from aiogram.filters.command import Command
from aiogram.filters.exception import ExceptionTypeFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.strategy import FSMStrategy
from aiogram.types import Message, User, ErrorEvent

TELEGRAM_API_TOKEN: str = env.get("TELEGRAM_API_TOKEN") # type: ignore
TELEGRAM_WHITELISTED_USERS: str = env.get("TELEGRAM_WHITELISTED_USERS") # type: ignore
OPENAI_API_KEY: str = env.get("OPENAI_API_KEY") # type: ignore

bot = Bot(token=TELEGRAM_API_TOKEN, parse_mode="Markdown")

dp = Dispatcher(
    storage=MemoryStorage(),
    fsm_strategy=FSMStrategy.USER_IN_CHAT)

router = Router(name="FormRouter")

dp.include_router(router)

class WhitelistedUsers(Filter):
    def __init__(self, ids: List[int]) -> None:
        self.ids : List[int] = ids

    async def __call__(self, message: Message) -> bool:
        user : User = message.from_user # type: ignore

        is_valid = any([id == user.id for id in self.ids])

        if not is_valid:
            raise PermissionError()

        return is_valid

def parse_user_ids(comma_separated: str) -> List[int]:
    return [int(value) for value in comma_separated.split(",")]

@router.errors(ExceptionTypeFilter(PermissionError))
async def access_error_handler(error_event: ErrorEvent) -> None:
    message : Message = error_event.update.message # type: ignore

    await message.reply(
        text="Access denied.",
        reply_markup=None)

@router.message(
    WhitelistedUsers(
        parse_user_ids(TELEGRAM_WHITELISTED_USERS)))
async def prompt_handler(message: Message) -> None:
    if (message.text is not None and len(message.text) > 0):
        answer = get_answer(message.text)
        await bot.send_message(message.chat.id, answer)

@router.message(Command("my_telegram_id"))
async def get_my_telegram_id(message: Message) -> None:
    user : User = message.from_user # type: ignore

    await message.reply(
        text=f"`{user.id}`",
        reply_markup=None)

def get_answer(prompt: str) -> str:
    openai.api_key = OPENAI_API_KEY

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=500)

    return response.choices[0].text # type: ignore

if __name__ == '__main__':
    dp.run_polling(bot)
