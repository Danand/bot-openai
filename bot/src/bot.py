import sys
import openai
import asyncio
import inspect
import time

import logging as log
import simplejson as json
import pkg_resources as packages

from os import environ
from typing import List, Dict, cast

from redis.asyncio.client import Redis

from environs import Env

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters.command import Command, CommandStart
from aiogram.filters.exception import ExceptionTypeFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.strategy import FSMStrategy
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.enums import BotCommandScopeType
from aiogram.enums.chat_action import ChatAction

from aiogram.types import (
    Message,
    ErrorEvent,
    BotCommand,
    BotCommandScopeChat,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    ForceReply,
    ReplyKeyboardRemove,
)

from filters import (
    IsNotCommand,
    IsNotReply,
    AuthorizedOnly,
    WhitelistedUsers,
    AddedUsers,
)

from aliases import SendMessageDelegate
from states import States
from utils import strip_markdown, split_string_to_batches

env = Env()

env.read_env(recurse=True)

LOG_LEVEL: int = env.log_level("LOG_LEVEL", default=log.INFO)

log.basicConfig(
    stream=sys.stdout,
    level=LOG_LEVEL,
)

# HACK: Suppress error caused by late initialization of `working_set`
working_set = [] if packages.working_set is None else packages.working_set

log.debug(f"Packages:\n{json.dumps([package.project_name for package in working_set], indent=True)}")
log.debug(f"Environment:\n{json.dumps(dict(environ.items()), indent=True)}")

OPENAI_DEFAULT_MAX_TOKENS_LIMIT: int = 4097

IS_IN_DOCKER: bool = env.bool("IS_IN_DOCKER", default=False)
TELEGRAM_API_TOKEN: str = env.str("TELEGRAM_API_TOKEN")
TELEGRAM_WHITELISTED_USERS: List[int] = cast(List[int], env.list("TELEGRAM_WHITELISTED_USERS", subcast=int))
OPENAI_API_KEY: str = env.str("OPENAI_API_KEY")
OPENAI_DEFAULT_MODEL: str = env.str("OPENAI_DEFAULT_MODEL", default="gpt-4o")
OPENAI_DEFAULT_TEMPERATURE: float = env.float("OPENAI_DEFAULT_TEMPERATURE", default=1.0)
OPENAI_DEFAULT_MAX_TOKENS: int = env.int("OPENAI_DEFAULT_MAX_TOKENS", default=OPENAI_DEFAULT_MAX_TOKENS_LIMIT)
OPENAI_DEFAULT_MAX_MESSAGES: int = env.int("OPENAI_DEFAULT_MAX_MESSAGES", default=6)
REDIS_HOST: str = env.str("REDIS_HOST", default="localhost") if IS_IN_DOCKER else "localhost"
REDIS_PORT: int = env.int("REDIS_PORT", default=6379)

env.seal()

log.debug(f"Environment parsed:\n{env.dump()}")

bot = Bot(
    token=TELEGRAM_API_TOKEN,
    default=DefaultBotProperties(
        parse_mode="Markdown",
    ),
)

log.info("Bot initialized")

redis = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
)

log.info("Redis initialized")

dispatcher = Dispatcher(
    storage=RedisStorage(redis),
    fsm_strategy=FSMStrategy.USER_IN_CHAT,
)

log.info("Dispatcher initialized")

router = Router()

dispatcher.include_router(router)

log.info("Router included into dispatcher")

send_message_delegates: List[SendMessageDelegate] = [
    lambda bot, chat_id, text: bot.send_message(chat_id, text, parse_mode="Markdown"),
    lambda bot, chat_id, text: bot.send_message(chat_id, text),
    lambda bot, chat_id, text: bot.send_message(chat_id, text, parse_mode="HTML"),
    lambda bot, chat_id, text: bot.send_message(chat_id, strip_markdown(text)),
    lambda bot, chat_id, text: bot.send_message(chat_id, f"```\n{text}\n```"),
]

@router.errors(ExceptionTypeFilter(PermissionError))
async def access_error_handler(error_event: ErrorEvent) -> None:
    message : Message = error_event.update.message # type: ignore

    await message.reply(
        text="Access denied.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

async def log_message(message: Message, state: FSMContext) -> None:
    if LOG_LEVEL > log.DEBUG:
        return

    current_state = await state.get_state()
    data = await state.get_data()

    model = data.get("model", OPENAI_DEFAULT_MODEL)
    temperature = float(data.get("temperature", OPENAI_DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", OPENAI_DEFAULT_MAX_TOKENS))
    max_messages = int(data.get("max_messages", OPENAI_DEFAULT_MAX_MESSAGES))

    messages: List[Dict[str, str]] = data.get("saved_messages", [])

    user_id = from_user.id if (from_user := message.from_user) is not None else None

    message_log_data = {
        "from_user": {
            "id": user_id,
        },
        "state": {
            "name": current_state,
        },
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_messages": max_messages,
        "messages": {
            "count": len(messages),
        },
    }

    log.debug(f"Message\n{json.dumps(message_log_data, indent=True)}")

@router.message(
    IsNotCommand(),
    IsNotReply(),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def prompt_handler(message: Message, state: FSMContext) -> None:
    await log_message(message, state)

    send_typing_task = asyncio.create_task(send_typing(message))
    send_answer_task = asyncio.create_task(send_prompt(message, state))

    concurrent_tasks = [send_typing_task, send_answer_task]

    done_tasks, _ = await asyncio.wait(
        fs=concurrent_tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    undone_tasks = [
        concurrent_task
        for concurrent_task in concurrent_tasks
        if concurrent_task not in done_tasks
    ]

    for undone_task in undone_tasks:
        undone_task.cancel()

async def send_message_with_retry(bot: Bot, chat_id: int, text: str) -> None:
    last_exception: BaseException | None = None

    for send_message_delegate in send_message_delegates:
        try:
            await send_message_delegate(bot, chat_id, text)
            return
        except BaseException as send_exception:
            log.error(f"Cannot parse answer:\n{text}")

            if "parse" not in str(send_exception):
                raise

            last_exception = send_exception

    raise Exception("Totally unknown exception") if last_exception is None else last_exception

async def send_prompt(message: Message, state: FSMContext) -> None:
    prompt = message.text

    if prompt is None or len(prompt) == 0:
        return

    data = await state.get_data()

    saved_messages: List[Dict[str, str]] = data.get("saved_messages", [])

    saved_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    await state.update_data(saved_messages=saved_messages)

    try:
        answer = await get_answer(prompt, state)

        if answer is None:
            return

        saved_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        max_messages = int(data.get("max_messages", OPENAI_DEFAULT_MAX_MESSAGES))

        if max_messages != 0 and len(saved_messages) > max_messages:
            saved_messages.pop(0)

        await state.update_data(saved_messages=saved_messages)

        if (len(answer.encode()) < 4096):
            await send_message_with_retry(bot, message.chat.id, answer)
        else:
            for batch in split_string_to_batches(answer, 2048):
                await send_message_with_retry(bot, message.chat.id, batch)

    except BaseException as exception:
        log.error(exception)

        text = f"Exception occurred:\n```log\n{exception}\n```"

        await bot.send_message(message.chat.id, text)

async def send_typing(message: Message) -> None:
    try:
        while True:
            await bot.send_chat_action(
                chat_id=message.chat.id,
                action=ChatAction.TYPING,
            )

            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass

@router.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.default)

    await message.reply(
        text="Hi! Write your prompt or check out available commands.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

    commands_non_authorized = [
        BotCommand(command="get_my_telegram_id", description="Get my Telegram ID"),
        BotCommand(command="start", description="Refresh commands"),
    ]

    commands_authorized = [
        BotCommand(command="set_temperature", description="Set temperature (creativity)"),
        BotCommand(command="set_model", description="Choose ChatGPT model"),
        BotCommand(command="reset_conversation", description="Reset conversation"),
    ]

    commands_elevated = [
        BotCommand(command="set_max_tokens", description="Set token limit"),
        BotCommand(command="set_max_messages", description="Set context limit."),
        BotCommand(command="add_user", description="Grant access to someone"),
        BotCommand(command="remove_user", description="Revoke access from someone"),
    ]

    commands_all = []

    try:
        await WhitelistedUsers(TELEGRAM_WHITELISTED_USERS)(message)
        commands_all.append(*commands_authorized)
        commands_all.append(*commands_elevated)
    except PermissionError:
        try:
            await AddedUsers(redis)(message)
            commands_all.append(*commands_authorized)
        except PermissionError:
            pass

    commands_all.append(*commands_non_authorized)

    await bot.set_my_commands(
        commands=commands_all,
        scope=BotCommandScopeChat(
            type=BotCommandScopeType.CHAT,
            chat_id=message.chat.id,
        ),
    )

@router.message(
    Command("get_my_telegram_id"),
)
async def get_my_telegram_id_handler(message: Message, state: FSMContext) -> None:
    user = message.from_user

    if user is not None:
        await message.reply(
            text=f"`{user.id}`",
            reply_markup=ReplyKeyboardRemove(
                remove_keyboard=True,
            ),
        )

@router.message(
    Command("set_model"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def set_model_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.model)

    models_all = openai.models.list()

    models_chat_completion = [
        model
        for model in models_all
    ]

    await message.reply(
        text="Choose OpenAI model.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=model.id, callback_data=model.id)]
                for model in models_chat_completion
            ],
        ),
    )

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
            text=f"Model changed to {model}",
        )

@router.message(
    Command("set_temperature"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
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

        buttons[row_index].append(
            InlineKeyboardButton(
                text=temperature_value,
                callback_data=temperature_value,
            ),
        )

    await message.reply(
        text="Choose creativity for responses.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )

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
            text=f"Temperature changed to {temperature}",
        )

@router.message(
    Command("set_max_tokens"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def set_max_tokens_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.max_tokens)

    await message.reply(
        text=f"Enter a number of token limit per response. {OPENAI_DEFAULT_MAX_TOKENS_LIMIT} is maximum.",
        reply_markup=ForceReply(
            force_reply=True,
            input_field_placeholder=str(OPENAI_DEFAULT_MAX_TOKENS),
        ),
    )

@router.message(States.max_tokens)
async def reply_max_tokens_handler(message: Message, state: FSMContext) -> None:
    max_tokens = message.text

    if max_tokens is None \
       or not max_tokens.isdigit() \
       or int(max_tokens) > OPENAI_DEFAULT_MAX_TOKENS_LIMIT:
        await message.reply(
            text=f"Enter a correct number of token limit per response. {OPENAI_DEFAULT_MAX_TOKENS_LIMIT} is maximum",
            reply_markup=ForceReply(
                force_reply=True,
                input_field_placeholder=str(OPENAI_DEFAULT_MAX_TOKENS),
            ),
        )

        return

    if max_tokens.strip() == "0":
        max_tokens = str(OPENAI_DEFAULT_MAX_TOKENS)

    await state.update_data(max_tokens=max_tokens)
    await state.set_state(States.default)

    await message.reply(
        text=f"Token limit changed to {max_tokens}.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

@router.message(
    Command("set_max_messages"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def set_max_messages_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.max_messages)

    await message.reply(
        text="Enter a number of context limit. 0 is for unlimited.",
        reply_markup=ForceReply(
            force_reply=True,
            input_field_placeholder=str(OPENAI_DEFAULT_MAX_MESSAGES),
        ),
    )

@router.message(States.max_tokens)
async def reply_max_messages_handler(message: Message, state: FSMContext) -> None:
    max_messages = message.text

    if max_messages is None or not max_messages.isdigit():
        await message.reply(
            text="Enter a correct number of context limit. 0 is for unlimited.",
            reply_markup=ForceReply(
                force_reply=True,
                input_field_placeholder=str(OPENAI_DEFAULT_MAX_MESSAGES),
            ),
        )

        return

    await state.update_data(max_messages=max_messages)
    await state.set_state(States.default)

    await message.reply(
        text=f"Context limit changed to {max_messages}.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

@router.message(
    Command("reset_conversation"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def reset_conversation_handler(message: Message, state: FSMContext) -> None:
    await state.update_data(saved_messages=[])

    await message.reply(
        text=f"Conversation forgotten.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

@router.message(
    Command("add_user"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def add_user_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.add_user)

    await message.reply(
        text="Share contact of user you want to grant access to this bot.",
        reply_markup=ForceReply(force_reply=True),
    )

@router.message(States.add_user)
async def reply_add_user_handler(message: Message, state: FSMContext) -> None:
    contact = message.contact

    if contact is None or contact.user_id is None:
        await message.reply(
            text=f"There is no valid contact of user.",
            reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
        )

        await state.set_state(States.default)

        return

    rpush_task = redis.rpush("users", str(contact.user_id))

    if inspect.isawaitable(rpush_task):
        await rpush_task

    await state.set_state(States.default)

    await message.reply(
        text=f"Access granted to user `{contact.user_id}`.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

@router.message(
    Command("remove_user"),
    AuthorizedOnly(redis, TELEGRAM_WHITELISTED_USERS),
)
async def remove_user_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(States.remove_user)

    await message.reply(
        text="Share contact of user you want revoke access to this bot.",
        reply_markup=ForceReply(force_reply=True),
    )

@router.message(States.remove_user)
async def reply_remove_user_handler(message: Message, state: FSMContext) -> None:
    contact = message.contact

    if contact is None or contact.user_id is None:
        await message.reply(
            text=f"There is no valid contact of user.",
            reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
        )

        await state.set_state(States.default)

        return

    lrem_task = redis.lrem(
        name="users",
        count=0,
        value=str(contact.user_id),
    )

    if inspect.isawaitable(lrem_task):
        await lrem_task

    await state.set_state(States.default)

    await message.reply(
        text=f"Access revoked from user `{contact.user_id}`.",
        reply_markup=ReplyKeyboardRemove(remove_keyboard=True),
    )

async def get_answer(prompt: str, state: FSMContext) -> str | None:
    openai.api_key = OPENAI_API_KEY

    data = await state.get_data()

    model = data.get("model", OPENAI_DEFAULT_MODEL)
    temperature = float(data.get("temperature", OPENAI_DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", OPENAI_DEFAULT_MAX_TOKENS))
    max_messages = int(data.get("max_messages", OPENAI_DEFAULT_MAX_MESSAGES))

    messages = data.get(
        "saved_messages",
        [
            {"role": "user", "content": prompt}
        ]
    )

    while True:
        try:
            response = openai.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=None if max_tokens == OPENAI_DEFAULT_MAX_TOKENS_LIMIT else max_tokens,
                messages=messages,
            )

            return response.choices[0].message.content

        except BaseException as answer_exception:
            if "maximum context length" in str(answer_exception):
                if len(messages) > 1:
                    messages.pop(0)
                else:
                    raise
            # Prevent flood:
            elif "tokens per min" in str(answer_exception):
                if len(messages) > max_messages:
                    while len(messages) > max_messages:
                        messages.pop(0)

                time.sleep(60)
            else:
                raise
        finally:
            await state.update_data(saved_messages=messages)

if __name__ == '__main__':
    log.info("About to run polling")
    dispatcher.run_polling(bot)
