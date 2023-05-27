from aiogram.fsm.state import State, StatesGroup

class States(StatesGroup):
    default = State()
    model = State()
    temperature = State()
    max_tokens = State()
    saved_messages = State()
    max_messages = State()
    add_user = State()
    remove_user = State()
