from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    sample_values: List[Any]


class DatasetObservation(BaseModel):
    task_id: int
    task_description: str
    goal: str
    columns: List[ColumnInfo]
    row_count: int
    duplicate_count: int
    step_count: int
    max_steps: int
    done: bool = False
    message: str = ""


class CleaningAction(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "One of: rename_column, rename_all_snake_case, fill_missing, "
            "drop_duplicates, convert_type, remove_outliers, submit"
        ),
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CleaningReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


# OpenEnv spec aliases
Observation = DatasetObservation
Action = CleaningAction
Reward = CleaningReward
