from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime


class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass(frozen=True)
class Order:
    action: Action
    timestamp: datetime