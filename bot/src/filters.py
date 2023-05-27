import inspect

from typing import List

from redis.asyncio.client import Redis

from aiogram.filters import Filter

from aiogram.types import (
    Message,
    User,
)

class WhitelistedUsers(Filter):
    def __init__(self, ids: List[int]) -> None:
        self.ids: List[int] = ids

    async def __call__(self, message: Message) -> bool:
        if message.from_user is None:
            return False

        user: User = message.from_user

        is_valid = any([id == user.id for id in self.ids])

        if not is_valid:
            raise PermissionError()

        return is_valid

class AddedUsers(Filter):
    def __init__(self, redis: Redis) -> None:
        self.redis: Redis = redis

    async def __call__(self, message: Message) -> bool:
        if message.from_user is None:
            raise PermissionError("Failed to detect you")

        user: User = message.from_user

        lrange_task = self.redis.lrange("users", 0, -1)

        if not inspect.isawaitable(lrange_task):
            raise PermissionError("Failed to read list of users")

        ids: List[int] = await lrange_task

        is_valid = any([int(id) == user.id for id in ids])

        if not is_valid:
            raise PermissionError()

        return is_valid

class AuthorizedOnly(Filter):
    def __init__(self, redis: Redis, whitelisted_ids: List[int]) -> None:
        self.redis: Redis = redis
        self.whitelisted_ids: List[int] = whitelisted_ids

    async def __call__(self, message: Message) -> bool:
        try:
            return await AddedUsers(self.redis).__call__(message)
        except PermissionError:
            return await WhitelistedUsers(self.whitelisted_ids).__call__(message)

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
