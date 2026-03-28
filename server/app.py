"""
DataCleaningEnv — OpenEnv-compliant FastAPI server.

Endpoints:
  GET  /           → health check
  GET  /tasks      → list all tasks
  GET  /state      → current environment state
  POST /reset      → reset environment for a task
  POST /step       → apply an action
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict

from environment import DataCleaningEnv, Action
from environment.tasks import TASKS

app = FastAPI(
    title="DataCleaningEnv",
    description=(
        "An OpenEnv environment for training AI agents on real-world "
        "CSV data cleaning tasks: rename columns, impute missing values, "
        "remove outliers, and deduplicate records."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per session)
env = DataCleaningEnv()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "data-cleaning-env",
        "version": "1.0.0",
        "status": "ok",
        "tasks": len(TASKS),
    }


@app.get("/tasks")
def get_tasks() -> list:
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
            "description": t["description"],
            "goal": t.get("goal", ""),
        }
        for t in TASKS.values()
    ]


@app.post("/reset")
def reset(task_id: int = 1) -> Dict[str, Any]:
    """Reset the environment and return the initial observation."""
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    """Apply a cleaning action and return the next observation, reward, done, info."""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return a lightweight snapshot of the current environment state."""
    return env.state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
