from typing import Dict, Type

from .q_learn import QLearnPlayer
from .learned_base import BaseLearnedPlayer

PLAYER_TYPES: Dict[str, Type[BaseLearnedPlayer]] = {
    "q_learn": QLearnPlayer
}
