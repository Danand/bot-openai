from os import environ as env
from typing import List

import openai

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Filter
from aiogram.filters.command import Command, CommandStart
from aiogram.filters.exception import ExceptionTypeFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.strategy import FSMStrategy
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.enums.chat_action import ChatAction

from aiogram.types import (
    Message,
    User,
    ErrorEvent,
    BotCommand,
    BotCommandScopeAllPrivateChats,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)

from redis.asyncio.client import Redis

TELEGRAM_API_TOKEN: str = env.get("TELEGRAM_API_TOKEN") # type: ignore
TELEGRAM_WHITELISTED_USERS: str = env.get("TELEGRAM_WHITELISTED_USERS") # type: ignore
OPENAI_API_KEY: str = env.get("OPENAI_API_KEY") # type: ignore
REDIS_HOST: str = env.get("REDIS_HOST", "localhost") # type: ignore
REDIS_PORT: int = int(env.get("REDIS_PORT", "6379")) # type: ignore
DEFAULT_MODEL: str = env.get("DEFAULT_MODEL", "gpt-3.5-turbo") # type: ignore
DEFAULT_TEMPERATURE: str = env.get("DEFAULT_TEMPERATURE", "1.0") # type: ignore

bot = Bot(token=TELEGRAM_API_TOKEN, parse_mode="Markdown")

redis = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT)

dispatcher = Dispatcher(
    storage=RedisStorage(redis),
    fsm_strategy=FSMStrategy.USER_IN_CHAT)

router = Router()

dispatcher.include_router(router)

class States(StatesGroup):
    default = State()
    model = State()
    temperature = State()
    max_tokens = State()
    remember_messages = State()

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
async def prompt_handler(message: Message, state: FSMContext) -> None:
    if (message.text is not None and len(message.text) > 0):
        await bot.send_chat_action(
            chat_id=message.chat.id,
            action=ChatAction.TYPING)

        try:
            answer = await get_answer(message.text, state)
            await bot.send_message(message.chat.id, answer)
        except Exception as exception:
            text = f"Exception occurred:\n```log\n{exception}\n```"
            await bot.send_message(message.chat.id, text)

@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.default)

    await message.reply("Hi! Write your prompt or check out available commands.")

    await bot.set_my_commands(commands=[
        BotCommand(command="get_my_telegram_id", description="Get my Telegram ID"),
        BotCommand(command="set_model", description="Choose ChatGPT model"),
        BotCommand(command="set_temperature", description="Set temperature (creativity)"),
        BotCommand(command="max_tokens", description="Set tokens limit"),
        BotCommand(command="reset_conversation", description="Reset conversation"),
    ], scope=BotCommandScopeAllPrivateChats(type="all_private_chats"))

@router.message(Command("get_my_telegram_id"))
async def get_my_telegram_id_handler(message: Message, state: FSMContext) -> None:
    user : User = message.from_user # type: ignore

    await message.reply(
        text=f"`{user.id}`",
        reply_markup=None)

@router.message(Command("set_model"))
async def set_model_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.model)

    models_all = openai.Model.list()["data"] # type: ignore
    models_chat_completion = [model for model in models_all if model.id.startswith("text-davinci") or model.id.startswith("gpt-")]

    await message.reply(# type: ignore
        text="Choose OpenAI model.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=model.id, callback_data=model.id)] for model in models_chat_completion]))

@router.callback_query(States.model)
async def callback_query_model_handler(callbackQuery: CallbackQuery, state: FSMContext) -> None:
    model = callbackQuery.data

    await state.update_data(model=model)
    await state.set_state(States.default)

    await callbackQuery.answer(f"Chosen model: {model}")

@router.message(Command("set_temperature"))
async def set_temperature_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.temperature)

    buttons: List[List[InlineKeyboardButton]] = []

    rows_amount = 5

    for row_index in range(rows_amount):
        buttons.append([])

    temperature_max = 2
    temperature_multiplier = 10
    range_offset = 1

    for temperature_index in range(range_offset, ((temperature_max * temperature_multiplier) + range_offset)):
        row_index = (temperature_index - range_offset) // rows_amount
        temperature_value = str(temperature_index / float(temperature_multiplier))

        buttons[row_index].append(InlineKeyboardButton(
            text=temperature_value,
            callback_data=temperature_value))

    await message.reply(
        text="Choose creativity for responses.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))

@router.callback_query(States.temperature)
async def callback_query_temperature_handler(callbackQuery: CallbackQuery, state: FSMContext) -> None:
    temperature = callbackQuery.data

    await state.update_data(temperature=temperature)
    await state.set_state(States.default)

    await callbackQuery.answer(f"Chosen temperature: {temperature}")

async def get_answer(prompt: str, state: FSMContext) -> str:
    openai.api_key = OPENAI_API_KEY

    data = await state.get_data()

    model = data.get("model", DEFAULT_MODEL)
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)

    response = await openai.ChatCompletion.acreate(
        model=model,
        temperature=float(temperature),
        messages=[{"role": "user", "content": prompt}])

    return response.choices[0].message.content # type: ignore

if __name__ == '__main__':
    dispatcher.run_polling(bot)
