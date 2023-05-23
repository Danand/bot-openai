import sys
import openai
import asyncio

import logging as log
import simplejson as json
import pkg_resources as packages

from os import environ as env
from typing import List, Dict
from distutils.util import strtobool

from redis.asyncio.client import Redis

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
    ForceReply,
    ReplyKeyboardRemove
)

LOG_LEVEL = env.get("LOG_LEVEL", "INFO")

log_level_int = int(getattr(log, LOG_LEVEL))

log.basicConfig(
    stream=sys.stdout,
    level=log_level_int)

log.debug(f"Packages:\n{json.dumps([package.project_name for package in packages.working_set], indent=True)}")
log.debug(f"Environment:\n{json.dumps(dict(env.items()), indent=True)}")

parse_bool = lambda value: bool(strtobool(str(value)))

IS_IN_DOCKER = parse_bool(env.get("IS_IN_DOCKER", "False"))
TELEGRAM_API_TOKEN: str = env.get("TELEGRAM_API_TOKEN") # type: ignore
TELEGRAM_WHITELISTED_USERS: str = env.get("TELEGRAM_WHITELISTED_USERS") # type: ignore
OPENAI_API_KEY: str = env.get("OPENAI_API_KEY") # type: ignore
OPENAI_DEFAULT_MODEL: str = env.get("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo") # type: ignore
OPENAI_DEFAULT_TEMPERATURE: str = env.get("OPENAI_DEFAULT_TEMPERATURE", "1.0") # type: ignore
OPENAI_DEFAULT_MAX_TOKENS: str = env.get("OPENAI_DEFAULT_MAX_TOKENS", "0") # type: ignore
REDIS_HOST: str = env.get("REDIS_HOST", "localhost") if IS_IN_DOCKER else "localhost" # type: ignore
REDIS_PORT: int = int(env.get("REDIS_PORT", "6379")) # type: ignore

bot = Bot(token=TELEGRAM_API_TOKEN, parse_mode="Markdown")

log.info("Bot initialized")

redis = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT)

log.info("Redis initialized")

dispatcher = Dispatcher(
    storage=RedisStorage(redis),
    fsm_strategy=FSMStrategy.USER_IN_CHAT)

log.info("Dispatcher initialized")

router = Router()

dispatcher.include_router(router)

log.info("Router included into dispatcher")

class States(StatesGroup):
    default = State()
    model = State()
    temperature = State()
    max_tokens = State()
    saved_messages = State()

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

class IsNotFromBot(Filter):
    async def __call__(self, message: Message) -> bool:
        return message.via_bot is None

class IsNotReply(Filter):
    async def __call__(self, message: Message) -> bool:
        return message.reply_to_message is None

def parse_user_ids(comma_separated: str) -> List[int]:
    return [int(value) for value in comma_separated.split(",")]

log.info("Filters defined")

@router.errors(ExceptionTypeFilter(PermissionError))
async def access_error_handler(error_event: ErrorEvent) -> None:
    message : Message = error_event.update.message # type: ignore

    await message.reply(
        text="Access denied.",
        reply_markup=None)

async def log_message(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    data = await state.get_data()

    model = data.get("model", OPENAI_DEFAULT_MODEL)
    temperature = float(data.get("temperature", OPENAI_DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", OPENAI_DEFAULT_MAX_TOKENS))

    messages: List[Dict[str, str]] = data.get("saved_messages", [])

    user_id = from_user.id if (from_user := message.from_user) is not None else None

    message_log_data = {
        "from_user": {
            "id": user_id
        },
        "state": {
            "name": current_state
        },
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": {
            "count": len(messages)
        },
    }

    log.debug(f"Message\n{json.dumps(message_log_data, indent=True)}")

@router.message(
    IsNotCommand(),
    IsNotReply(),
    WhitelistedUsers(
        parse_user_ids(TELEGRAM_WHITELISTED_USERS)))
async def prompt_handler(message: Message, state: FSMContext) -> None:
    await log_message(message, state)

    send_typing_task = asyncio.create_task(send_typing(message))
    send_answer_task = asyncio.create_task(send_prompt(message, state))

    concurrent_tasks = [send_typing_task, send_answer_task]

    done_tasks, _ = await asyncio.wait(concurrent_tasks, return_when=asyncio.FIRST_COMPLETED)

    undone_tasks = [concurrent_task for concurrent_task in concurrent_tasks if concurrent_task not in done_tasks]

    for undone_task in undone_tasks:
        undone_task.cancel()

async def send_prompt(message: Message, state: FSMContext) -> None:
    prompt = message.text

    if (prompt is None or len(prompt) == 0):
        return

    data = await state.get_data()
    saved_messages: List[Dict[str, str]] = data.get("saved_messages", [])

    saved_messages.append({"role": "user", "content": prompt})
    await state.update_data(saved_messages=saved_messages)

    try:
        answer = await get_answer(prompt, state)

        await bot.send_message(message.chat.id, answer, parse_mode="Markdown")

        saved_messages.append({"role": "assistant", "content": answer})
        await state.update_data(saved_messages=saved_messages)

    except Exception as exception:
        log.error(exception)
        text = f"Exception occurred:\n```log\n{exception}\n```"
        await bot.send_message(message.chat.id, text)

async def send_typing(message: Message) -> None:
    try:
        while True:
            await bot.send_chat_action(
                chat_id=message.chat.id,
                action=ChatAction.TYPING)

            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass

@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.default)

    await message.reply("Hi! Write your prompt or check out available commands.")

    await bot.set_my_commands(commands=[
        BotCommand(command="reset_conversation", description="Reset conversation"),
        BotCommand(command="set_temperature", description="Set temperature (creativity)"),
        BotCommand(command="set_model", description="Choose ChatGPT model"),
        BotCommand(command="set_max_tokens", description="Set token limit"),
        BotCommand(command="get_my_telegram_id", description="Get my Telegram ID"),
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

    await message.reply(
        text="Choose OpenAI model.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=model.id, callback_data=model.id)] for model in models_chat_completion]))

@router.callback_query(States.model)
async def callback_query_model_handler(callback_query: CallbackQuery, state: FSMContext) -> None:
    model = callback_query.data

    await state.update_data(model=model)
    await state.set_state(States.default)

    await callback_query.answer(f"Chosen model: {model}")

    message = callback_query.message

    if message is not None:
        await bot.send_message(
            chat_id=message.chat.id,
            text=f"Model changed to {model}")

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
async def callback_query_temperature_handler(callback_query: CallbackQuery, state: FSMContext) -> None:
    temperature = callback_query.data

    await state.update_data(temperature=temperature)
    await state.set_state(States.default)

    await callback_query.answer(f"Chosen temperature: {temperature}")

    message = callback_query.message

    if message is not None:
        await bot.send_message(
            chat_id=message.chat.id,
            text=f"Temperature changed to {temperature}")

@router.message(Command("set_max_tokens"))
async def set_max_tokens_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.max_tokens)

    await message.reply(
        text="Enter a number of token limit per response. 0 is for unlimited.",
        reply_markup=ForceReply(
            force_reply=True,
            input_field_placeholder=OPENAI_DEFAULT_MAX_TOKENS))

@router.message(States.max_tokens)
async def reply_max_tokens_handler(message: Message, state: FSMContext) -> None:
    max_tokens = message.text

    if max_tokens is None or not max_tokens.isdigit():
        await message.reply(
            text="Enter a correct number of token limit per response. 0 is for unlimited.",
            reply_markup=ForceReply(
                force_reply=True,
                input_field_placeholder=OPENAI_DEFAULT_MAX_TOKENS))

        return

    await state.update_data(max_tokens=max_tokens)
    await state.set_state(States.default)

    await message.reply(
        text=f"Token limit changed to {max_tokens}.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True))

@router.message(Command("reset_conversation"))
async def reset_conversation_handler(message: Message, state: FSMContext) -> None:
    await state.update_data(saved_messages=[])
    await message.reply(f"Conversation forgotten.")

async def get_answer(prompt: str, state: FSMContext) -> str:
    openai.api_key = OPENAI_API_KEY

    data = await state.get_data()

    model = data.get("model", OPENAI_DEFAULT_MODEL)
    temperature = float(data.get("temperature", OPENAI_DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", OPENAI_DEFAULT_MAX_TOKENS))

    messages: List[Dict[str, str]] = data.get("saved_messages", [{"role": "user", "content": prompt}])

    if max_tokens == 0:
        response = await openai.ChatCompletion.acreate(
            model=model,
            temperature=temperature,
            messages=messages)
    else:
        response = await openai.ChatCompletion.acreate(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages)

    return response.choices[0].message.content # type: ignore

if __name__ == '__main__':
    log.info("About to run polling")
    dispatcher.run_polling(bot)
