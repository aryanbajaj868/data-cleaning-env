import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple

from .models import Observation, Action, Reward, ColumnInfo
from .tasks import TASKS
from .graders import GRADERS


class DataCleaningEnv:
    """
    OpenEnv-compliant environment for real-world data cleaning tasks.

    An AI agent receives observations describing a messy dataset and must
    apply structured cleaning actions to achieve full data quality.

    Observation  → DatasetObservation (column stats, missing counts, task goal)
    Action       → CleaningAction     (action_type + parameters dict)
    Reward       → CleaningReward     (0.0–1.0 with per-criterion breakdown)
    """

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._task_id: int = 1
        self._step_count: int = 0
        self._done: bool = False
        self._message: str = ""

    # ─────────────────────────────────────────
    # OpenEnv interface
    # ─────────────────────────────────────────

    def reset(self, task_id: int = 1) -> Observation:
        """Reset the environment for the given task and return the initial observation."""
        if task_id not in TASKS:
            raise ValueError(
                f"Task {task_id} not found. Available tasks: {sorted(TASKS.keys())}"
            )
        self._task_id = task_id
        self._df = TASKS[task_id]["get_data"]()
        self._step_count = 0
        self._done = False
        self._message = "Environment reset. Start cleaning!"
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action to the dataset.
        Returns (observation, reward, done, info).
        """
        if self._done:
            obs = self._build_observation()
            obs.message = "Episode already finished — call reset() to start over."
            return obs, Reward(value=0.0, message="Episode already done."), True, {}

        task = TASKS[self._task_id]
        self._step_count += 1
        info: Dict[str, Any] = {}

        try:
            self._apply_action(action)
            self._message = f"Action '{action.action_type}' applied successfully."
        except Exception as exc:
            self._message = f"Action failed: {exc}"
            info["error"] = str(exc)
            # Small penalty for invalid actions
            current_score, breakdown = GRADERS[self._task_id](self._df)
            penalised = round(max(0.0, current_score - 0.05), 3)
            reward = Reward(value=penalised, breakdown=breakdown, message=self._message)
            return self._build_observation(), reward, self._done, info

        # Score current state
        score, breakdown = GRADERS[self._task_id](self._df)

        # Determine done
        if action.action_type == "submit" or score >= 1.0:
            self._done = True
        if self._step_count >= task["max_steps"]:
            self._done = True
            self._message += " Max steps reached."

        reward = Reward(value=score, breakdown=breakdown, message=self._message)
        return self._build_observation(), reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of the current environment state."""
        current_score = 0.0
        if self._df is not None:
            current_score, _ = GRADERS[self._task_id](self._df)
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "done": self._done,
            "row_count": len(self._df) if self._df is not None else 0,
            "columns": list(self._df.columns) if self._df is not None else [],
            "current_score": round(current_score, 3),
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _build_observation(self) -> Observation:
        task = TASKS[self._task_id]
        columns = []
        dup_count = 0

        if self._df is not None:
            for col in self._df.columns:
                sample = self._df[col].dropna().head(3).tolist()
                columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=str(self._df[col].dtype),
                        missing_count=int(self._df[col].isna().sum()),
                        sample_values=[str(v) for v in sample],
                    )
                )
            dup_count = int(self._df.duplicated().sum())

        return Observation(
            task_id=self._task_id,
            task_description=task["description"],
            goal=task.get("goal", ""),
            columns=columns,
            row_count=len(self._df) if self._df is not None else 0,
            duplicate_count=dup_count,
            step_count=self._step_count,
            max_steps=task["max_steps"],
            done=self._done,
            message=self._message,
        )

    def _apply_action(self, action: Action) -> None:  # noqa: C901
        """Dispatch and apply a CleaningAction to self._df."""
        p = action.parameters

        # ── rename a single column ──────────────────────────────────────────
        if action.action_type == "rename_column":
            old, new = p.get("old_name"), p.get("new_name")
            if old not in self._df.columns:
                raise ValueError(
                    f"Column '{old}' not found. Available: {list(self._df.columns)}"
                )
            self._df.rename(columns={old: new}, inplace=True)

        # ── auto-rename ALL columns to snake_case ───────────────────────────
        elif action.action_type == "rename_all_snake_case":
            import re
            mapping: Dict[str, str] = {}
            for col in self._df.columns:
                new_col = col.strip()
                # Handle camelCase / PascalCase → insert underscore before uppercase runs
                new_col = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', new_col)
                new_col = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', new_col)
                new_col = new_col.lower()
                for ch in (" ", "-", ".", "/"):
                    new_col = new_col.replace(ch, "_")
                while "__" in new_col:
                    new_col = new_col.replace("__", "_")
                new_col = new_col.strip("_")
                mapping[col] = new_col
            self._df.rename(columns=mapping, inplace=True)

        # ── fill missing values ──────────────────────────────────────────────
        elif action.action_type == "fill_missing":
            col = p.get("column")
            strategy = p.get("strategy", "value")
            if col not in self._df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(self._df.columns)}"
                )
            if strategy == "mean":
                fill_val = self._df[col].mean()
            elif strategy == "median":
                fill_val = self._df[col].median()
            elif strategy == "mode":
                fill_val = self._df[col].mode().iloc[0]
            else:  # "value"
                fill_val = p.get("value")
                if fill_val is None:
                    raise ValueError("'value' key required when strategy='value'.")
            self._df[col] = self._df[col].fillna(fill_val)

        # ── drop duplicates ──────────────────────────────────────────────────
        elif action.action_type == "drop_duplicates":
            subset = p.get("subset")  # list of cols or None
            self._df.drop_duplicates(subset=subset, keep="first", inplace=True)
            self._df.reset_index(drop=True, inplace=True)

        # ── convert column dtype ──────────────────────────────────────────────
        elif action.action_type == "convert_type":
            col = p.get("column")
            dtype = p.get("dtype")
            if col not in self._df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(self._df.columns)}"
                )
            if dtype == "float":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
            elif dtype == "int":
                self._df[col] = (
                    pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
                )
            elif dtype == "str":
                self._df[col] = self._df[col].astype(str)
            else:
                raise ValueError(f"Unsupported dtype '{dtype}'. Use float, int, or str.")

        # ── remove / cap outliers ─────────────────────────────────────────────
        elif action.action_type == "remove_outliers":
            col = p.get("column")
            method = p.get("method", "iqr")
            if col not in self._df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(self._df.columns)}"
                )
            if method == "iqr":
                q1 = self._df[col].quantile(0.25)
                q3 = self._df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (self._df[col] < lower) | (self._df[col] > upper)
                fill_strategy = p.get("fill", "median")
                if fill_strategy == "median":
                    replacement = self._df[col].median()
                elif fill_strategy == "mean":
                    replacement = self._df[col].mean()
                else:
                    replacement = p.get("value", self._df[col].median())
                self._df.loc[outlier_mask, col] = replacement
            elif method == "cap":
                max_val = p.get("max_value")
                min_val = p.get("min_value")
                if max_val is not None:
                    self._df[col] = self._df[col].clip(upper=float(max_val))
                if min_val is not None:
                    self._df[col] = self._df[col].clip(lower=float(min_val))
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'cap'.")

        # ── submit ────────────────────────────────────────────────────────────
        elif action.action_type == "submit":
            pass  # Triggers done in step()

        else:
            raise ValueError(
                f"Unknown action type: '{action.action_type}'. "
                f"Valid types: rename_column, rename_all_snake_case, fill_missing, "
                f"drop_duplicates, convert_type, remove_outliers, submit"
            )