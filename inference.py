"""
DataCleaningEnv — Baseline Inference Script
"""

import json
import os
import sys

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy-key")
HF_TOKEN: str = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ─────────────────────────────────────────────────────────────────────────────
# Safe OpenAI client initialisation — handles all httpx versions
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_client():
    from openai import OpenAI
    import httpx

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(var, None)

    try:
        return OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(mounts={}, timeout=60.0))
    except TypeError:
        pass

    try:
        return OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxies={}, timeout=60.0))
    except TypeError:
        pass

    return OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent. You receive observations
describing a messy dataset and must apply structured actions to clean it.

Available action types and their parameters:
  rename_column          {"old_name": "...", "new_name": "..."}
  rename_all_snake_case  {}
  fill_missing           {"column": "...", "strategy": "mean|median|mode|value", "value": <optional>}
  drop_duplicates        {"subset": ["col1", "col2"]}
  convert_type           {"column": "...", "dtype": "float|int|str"}
  remove_outliers        {"column": "...", "method": "iqr|cap", "fill": "median|mean"}
  submit                 {}

Strategy:
  1. Fix column names first (use rename_all_snake_case).
  2. Convert types (e.g. salary strings to float).
  3. Fill all missing values.
  4. Drop duplicates.
  5. Remove outliers.
  6. Call submit when done.

Always respond with ONLY valid JSON:
{"action_type": "...", "parameters": {...}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_on_task(task_id: int, client):
    resp = requests.post(
        f"{API_BASE_URL}/reset", params={"task_id": task_id}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()

    max_steps = obs["max_steps"]
    conversation = []
    final_score = 0.0
    steps_taken = 0

    for step_num in range(max_steps):
        obs_text = json.dumps(obs, indent=2)
        conversation.append({"role": "user", "content": f"Current observation:\n{obs_text}"})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                temperature=0.0,
                max_tokens=300,
            )
            raw = completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"WARN: LLM call failed at step {step_num + 1}: {exc}", flush=True)
            continue

        conversation.append({"role": "assistant", "content": raw})

        try:
            clean_raw = raw
            if "```" in clean_raw:
                clean_raw = clean_raw.split("```")[1]
                if clean_raw.startswith("json\n"):
                    clean_raw = clean_raw[5:]
            action_dict = json.loads(clean_raw)
        except (json.JSONDecodeError, IndexError):
            continue

        steps_taken = step_num + 1

        # ── STEP log — required format with square brackets ──
        print(f"[STEP] step={steps_taken} action={action_dict.get('action_type')}", flush=True)

        try:
            step_resp = requests.post(
                f"{API_BASE_URL}/step",
                json=action_dict,
                headers=HEADERS,
                timeout=30,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as exc:
            print(f"WARN: step request failed: {exc}", flush=True)
            break

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        final_score = float(reward["value"])

        print(f"[STEP] step={steps_taken} reward={final_score}", flush=True)

        if done:
            break

    return final_score, steps_taken


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        client = make_openai_client()
        print("INFO: OpenAI client initialised successfully", flush=True)
    except Exception as exc:
        print(f"ERROR: OpenAI client init failed — {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)

    try:
        ping = requests.get(f"{API_BASE_URL}/", headers=HEADERS, timeout=15)
        ping.raise_for_status()
        print(f"INFO: Environment reachable at {API_BASE_URL}", flush=True)
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {API_BASE_URL}: {exc}", flush=True)
        sys.exit(1)

    scores = {}

    for task_id in [1, 2, 3]:
        # ── START log — required format with square brackets ──
        print(f"[START] task={task_id}", flush=True)

        try:
            score, steps = run_agent_on_task(task_id, client)
        except Exception as exc:
            print(f"ERROR: task={task_id} crashed — {type(exc).__name__}: {exc}", flush=True)
            score, steps = 0.0, 0

        scores[task_id] = score

        # ── END log — required format with square brackets + steps count ──
        print(f"[END] task={task_id} score={score} steps={steps}", flush=True)

    avg = sum(scores.values()) / len(scores)
    print(f"DONE average_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()