from os import environ as env
from typing import List

import openai

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Filter
from aiogram.filters.command import Command, CommandStart
from aiogram.filters.exception import ExceptionTypeFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.strategy import FSMStrategy
from aiogram.types import Message, User, ErrorEvent, BotCommand, BotCommandScopeAllPrivateChats
from aiogram.enums.chat_action import ChatAction

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
class IsNotCommand(Filter):
    async def __call__(self, message: Message) -> bool:
        if message.text is None:
            return False

        return not message.text.startswith("/")

def parse_user_ids(comma_separated: str) -> List[int]:
    return [int(value) for value in comma_separated.split(",")]

@router.errors(ExceptionTypeFilter(PermissionError))
async def access_error_handler(error_event: ErrorEvent) -> None:
    message : Message = error_event.update.message # type: ignore

    await message.reply(
        text="Access denied.",
        reply_markup=None)

@router.message(
    IsNotCommand(),
    WhitelistedUsers(
        parse_user_ids(TELEGRAM_WHITELISTED_USERS)))
async def prompt_handler(message: Message) -> None:
    if (message.text is not None and len(message.text) > 0):
        await bot.send_chat_action(
            chat_id=message.chat.id,
            action=ChatAction.TYPING)

        answer = get_answer(message.text)

        await bot.send_message(message.chat.id, answer)

@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.reply("Hi! Write your prompt or check out available commands.")

    await bot.set_my_commands(commands=[
        BotCommand(command="my_telegram_id", description="Get my Telegram ID"),
    ], scope=BotCommandScopeAllPrivateChats(type="all_private_chats"))

@router.message(Command("my_telegram_id"))
async def my_telegram_id_handler(message: Message) -> None:
    user : User = message.from_user # type: ignore

    await message.reply(
        text=f"`{user.id}`",
        reply_markup=None)

def get_answer(prompt: str) -> str:
    openai.api_key = OPENAI_API_KEY

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content # type: ignore

if __name__ == '__main__':
    dp.run_polling(bot)
