from enum import Enum
from typing import Union, Literal


class Guards(Enum):
    INSAMPLE_GTE_CONDITION = "insample >= x"
    INSAMPLE_LT_CONDITION = "insample < x"
    TRUE_CONDITION = "true"
    FALSE_CONDITION = "false"


class Outputs(Enum):
    INSAMPLE_OUTPUT = "insample"
    INSAMPLE_PRIME_OUTPUT = "insample'"
    EMPTY_OUTPUT = ""
    BOT = '⊥'
    TOP = '⊤'


OUTPUT_ALPHABET = {Outputs.EMPTY_OUTPUT, Outputs.BOT, Outputs.TOP}

# GuardType = Literal[
#     Guards.INSAMPLE_GTE_CONDITION,
#     Guards.INSAMPLE_LT_CONDITION,
#     Guards.TRUE_CONDITION
# ]
#
# OutputType = Literal[
#     Outputs.INSAMPLE_OUTPUT,
#     Outputs.INSAMPLE_PRIME_OUTPUT,
#     Outputs.EMPTY_OUTPUT,
#     Outputs.BOT,
#     Outputs.TOP
# ]
