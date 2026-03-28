"""
DataCleaningEnv — Baseline Inference Script

Runs an LLM agent (via OpenAI-compatible client) against all 3 tasks
and reports reproducible baseline scores.

Environment variables required:
  API_BASE_URL   The base URL of the running DataCleaningEnv HF Space
  MODEL_NAME     The model identifier to use (e.g. gpt-4o-mini)
  OPENAI_API_KEY Your OpenAI / compatible API key
  HF_TOKEN       Your Hugging Face token (used for authenticated Space access)
"""

import json
import os
import sys
import time

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# Extra headers for HF Space authentication (if HF_TOKEN is set)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent. You receive observations
describing a messy dataset and must apply structured actions to clean it.

Available action types and their parameters:
  rename_column          {"old_name": "...", "new_name": "..."}
  rename_all_snake_case  {}   ← renames every column to snake_case in one step
  fill_missing           {"column": "...", "strategy": "mean|median|mode|value", "value": <optional>}
  drop_duplicates        {"subset": ["col1", "col2"]}  or  {"subset": null}
  convert_type           {"column": "...", "dtype": "float|int|str"}
  remove_outliers        {"column": "...", "method": "iqr|cap", "fill": "median|mean",
                          "max_value": <optional>, "min_value": <optional>}
  submit                 {}   ← use when cleaning is complete

Strategy:
  1. Fix column names first (use rename_all_snake_case for speed).
  2. Convert types (e.g. salary strings to float).
  3. Fill all missing values.
  4. Drop duplicates.
  5. Remove outliers.
  6. Call submit when done.

Always respond with ONLY valid JSON — no markdown, no explanation:
{"action_type": "...", "parameters": {...}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_on_task(task_id: int) -> float:
    task_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    print(f"\n{'=' * 60}")
    print(f"  TASK {task_id}  ({task_labels.get(task_id, '?')})")
    print("=" * 60)

    # Reset environment
    resp = requests.post(
        f"{API_BASE_URL}/reset", params={"task_id": task_id}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()

    print(f"Description : {obs['task_description']}")
    print(f"Goal        : {obs['goal']}")
    print(f"Max steps   : {obs['max_steps']}\n")

    conversation: list = []
    final_score: float = 0.0
    max_steps: int = obs["max_steps"]

    for step_num in range(max_steps):
        # Build user message from current observation
        obs_text = json.dumps(obs, indent=2)
        conversation.append({"role": "user", "content": f"Current observation:\n{obs_text}"})

        # Ask the model for the next action
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            temperature=0.0,
            max_tokens=300,
        )
        raw = completion.choices[0].message.content.strip()
        conversation.append({"role": "assistant", "content": raw})

        # Parse the action JSON
        try:
            clean_raw = raw
            if "```" in clean_raw:
                clean_raw = clean_raw.split("```")[1]
                if clean_raw.startswith("json\n"):
                    clean_raw = clean_raw[5:]
            action_dict = json.loads(clean_raw)
        except (json.JSONDecodeError, IndexError) as exc:
            print(f"  Step {step_num + 1:02d} | PARSE ERROR: {exc} | Raw: {raw[:80]}")
            continue

        action_type = action_dict.get("action_type", "?")
        params = action_dict.get("parameters", {})
        print(f"  Step {step_num + 1:02d} | {action_type} | params={json.dumps(params)}")

        # Send action to environment
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json=action_dict,
            headers=HEADERS,
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        final_score = float(reward["value"])

        breakdown_str = ", ".join(f"{k}={v}" for k, v in reward.get("breakdown", {}).items())
        print(f"          ↳ reward={final_score:.3f}  [{breakdown_str}]")

        if result.get("info", {}).get("error"):
            print(f"          ⚠ {result['info']['error']}")

        if done:
            print(f"\n  Episode finished after {step_num + 1} step(s).")
            break

    print(f"\n  ✓ Task {task_id} Final Score: {final_score:.3f}")
    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  DataCleaningEnv — Baseline Inference")
    print("=" * 60)
    print(f"  Environment : {API_BASE_URL}")
    print(f"  Model       : {MODEL_NAME}")

    # Verify environment is reachable
    try:
        ping = requests.get(f"{API_BASE_URL}/", headers=HEADERS, timeout=15)
        ping.raise_for_status()
        info = ping.json()
        print(f"  Status      : {info.get('status', 'unknown')}")
    except Exception as exc:
        print(f"\n  ERROR: Cannot reach environment at {API_BASE_URL}: {exc}")
        sys.exit(1)

    scores: dict = {}
    for task_id in [1, 2, 3]:
        scores[task_id] = run_agent_on_task(task_id)

    # Summary
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    labels = {1: "Easy  ", 2: "Medium", 3: "Hard  "}
    for tid, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  Task {tid} ({labels[tid]}): {score:.3f}  |{bar}|")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score : {avg:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
