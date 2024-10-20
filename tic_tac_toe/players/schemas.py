from typing import Dict
from pydantic import BaseModel, Field


class TDSettings(BaseModel):
    """
    Settings for a general TD-Learner Player.
    """
    random_seed: int = 9876
    epsilon_greedy: bool = True
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )
    default_q: float = 0.0
    discount_rate: float = Field(
        default=1.0,
        le=1.0,
        gt=0.0,
    )
    step_size: float = Field(
        default=0.2,
        le=1.0,
        gt=0.0,
    )


class TabularPolicy(BaseModel):
    """
    A tabular policy.
    """
    states: Dict[str, Dict[str, float]] = Field(
        description="Map of state -> (action -> q_value)"
    )
