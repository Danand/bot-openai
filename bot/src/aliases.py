from typing import (
    Callable,
    Coroutine,
    Any,
)

from aiogram import Bot

from aiogram.types import (
    Message,
)

SendMessageDelegate = Callable[[Bot, int, str], Coroutine[Any, Any, Message]]
