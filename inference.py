"""
DataCleaningEnv — Baseline Inference Script

Runs an LLM agent (via OpenAI-compatible client) against all 3 tasks
and reports reproducible baseline scores.

Strict Hackathon Variable Requirements:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
"""

import json
import os
import sys
import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ─────────────────────────────────────────────────────────────────────────────

# 1. Strict Hackathon Variables for the LLM
API_BASE_URL: str = os.environ.get("API_BASE_URL")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# 2. Your Environment URL (Where the FastAPI app lives)
# Defaults to your deployed Hugging Face Space, but can be overridden for local testing.
ENV_URL: str = os.environ.get("ENV_URL", "https://aryanbajaj868-data-cleaning-env.hf.space").rstrip("/")

# Initialize the OpenAI Client strictly using the hackathon requirements
client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key", 
    base_url=API_BASE_URL
)

# Extra headers for HF Space authentication if needed
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ─────────────────────────────────────────────────────────────────────────────
# Agent System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent. You receive observations
describing a messy dataset and must apply structured actions to clean it.

Available action types and their parameters:
  rename_column          {"old_name": "...", "new_name": "..."}
  rename_all_snake_case  {}   ← renames every column to snake_case in one step
  fill_missing           {"column": "...", "strategy": "mean|median|mode|value", "value": "optional"}
  drop_duplicates        {"subset": ["col1", "col2"]}  or  {"subset": null}
  convert_type           {"column": "...", "dtype": "float|int|str"}
  remove_outliers        {"column": "...", "method": "iqr|cap", "fill": "median|mean", "max_value": 1000.0}
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
# Agent Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_on_task(task_id: int) -> float:
    task_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    print(f"\n{'=' * 60}")
    print(f"  TASK {task_id}  ({task_labels.get(task_id, '?')})")
    print("=" * 60)

    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset?task_id={task_id}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"  ❌ Failed to reset environment: {e}")
        return 0.0

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
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                temperature=0.0,
                max_tokens=300,
            )
            raw = completion.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": raw})
        except Exception as e:
            print(f"  ❌ LLM API Error: {e}")
            break

        # Parse the action JSON
        try:
            clean_raw = raw
            if "```" in clean_raw:
                clean_raw = clean_raw.split("```")[1]
                if clean_raw.startswith("json\n"):
                    clean_raw = clean_raw[5:]
            action_dict = json.loads(clean_raw)
        except (json.JSONDecodeError, IndexError) as exc:
            print(f"  Step {step_num + 1:02d} | ⚠ PARSE ERROR: {exc} | Raw output: {raw[:80]}")
            continue

        action_type = action_dict.get("action_type", "?")
        params = action_dict.get("parameters", {})
        print(f"  Step {step_num + 1:02d} | 🤖 {action_type} | params={json.dumps(params)}")

        # Send action to environment
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action_dict, headers=HEADERS, timeout=30)
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            print(f"  ❌ Environment API Error during step: {e}")
            break

        obs = result.get("observation", {})
        reward = result.get("reward", {"value": 0.0})
        done = result.get("done", True)
        final_score = float(reward.get("value", 0.0))

        breakdown_str = ", ".join(f"{k}={v}" for k, v in reward.get("breakdown", {}).items())
        print(f"          ↳ reward={final_score:.3f}  [{breakdown_str}]")

        if result.get("info", {}).get("error"):
            print(f"          ⚠ Env Error: {result['info']['error']}")

        if done:
            print(f"\n  🏁 Episode finished after {step_num + 1} step(s).")
            break

    print(f"\n  ✓ Task {task_id} Final Score: {final_score:.3f}")
    return final_score

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  DataCleaningEnv — Baseline Inference")
    print("=" * 60)
    print(f"  Environment Target : {ENV_URL}")
    print(f"  LLM API Base URL   : {API_BASE_URL if API_BASE_URL else 'OpenAI Default'}")
    print(f"  Model              : {MODEL_NAME}")

    # Verify environment is reachable before starting
    try:
        ping = requests.get(f"{ENV_URL}/", headers=HEADERS, timeout=15)
        ping.raise_for_status()
        info = ping.json()
        print(f"  Status             : {info.get('status', 'unknown')}")
    except Exception as exc:
        print(f"\n  ❌ FATAL ERROR: Cannot reach environment at {ENV_URL}: {exc}")
        print("  Make sure your Hugging Face space is 'Running' and not 'Sleeping'.")
        sys.exit(1)

    # Run the tasks
    scores: dict = {}
    for task_id in [1, 2, 3]:
        scores[task_id] = run_agent_on_task(task_id)

    # Print Summary Report
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