"""
DataCleaningEnv — Improved Inference Script
"""

import json
import os
import sys

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LLM_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
API_KEY: str = os.environ.get("API_KEY", "dummy-key")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client — routed through hackathon LiteLLM proxy
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_client() -> OpenAI:
    import httpx
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(var, None)

    print(f"INFO: LLM proxy base_url={LLM_BASE_URL}", flush=True)

    for kwargs in ({"mounts": {}}, {"proxies": {}}, {}):
        try:
            return OpenAI(
                api_key=API_KEY,
                base_url=LLM_BASE_URL,
                http_client=httpx.Client(timeout=60.0, **kwargs),
            )
        except TypeError:
            continue
    return OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt — improved
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data cleaning agent working step-by-step on a messy dataset.

Available actions (respond ONLY with valid JSON, no explanation):
  {"action_type": "rename_all_snake_case", "parameters": {}}
  {"action_type": "rename_column", "parameters": {"old_name": "...", "new_name": "..."}}
  {"action_type": "convert_type", "parameters": {"column": "...", "dtype": "float|int|str"}}
  {"action_type": "fill_missing", "parameters": {"column": "...", "strategy": "mean|median|mode|value", "value": <optional>}}
  {"action_type": "drop_duplicates", "parameters": {"subset": ["col1", "col2"]}}
  {"action_type": "remove_outliers", "parameters": {"column": "...", "method": "iqr|cap", "fill": "median|mean"}}
  {"action_type": "submit", "parameters": {}}

STRICT RULES:
  1. FIRST action must always be rename_all_snake_case (fixes all column names at once).
  2. THEN convert any columns that look like numbers but are stored as strings (e.g. "$1,200" → float).
  3. THEN fill ALL missing values — use median for numeric columns, mode for categorical.
  4. THEN drop_duplicates using all columns as subset.
  5. THEN remove_outliers on every numeric column using method=iqr, fill=median.
  6. ONLY call submit when the observation shows zero missing values, zero duplicates, and all types are correct.
  7. NEVER call submit if there are still issues remaining in the observation.
  8. If the last action had no effect (reward did not improve), try a DIFFERENT action on a DIFFERENT column.
  9. Never repeat the exact same action twice in a row.

Read the observation carefully — it tells you exactly which columns have issues.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, messages: list) -> str:
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=400,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"WARN: LLM attempt {attempt+1} failed: {exc}", flush=True)
    return ""


def parse_action(raw: str) -> dict | None:
    try:
        clean = raw
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json\n"):
                clean = clean[5:]
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return None


def obs_has_issues(obs: dict) -> bool:
    """Return True if the observation still reports problems."""
    try:
        issues = obs.get("issues", {})
        if isinstance(issues, dict):
            return any(v for v in issues.values())
        stats = obs.get("stats", {})
        missing = stats.get("missing_values", {})
        if any(v > 0 for v in missing.values()):
            return True
        if stats.get("duplicate_rows", 0) > 0:
            return True
        return False
    except Exception:
        return True  # assume issues exist if we can't parse

# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_on_task(task_id: int, client: OpenAI):
    resp = requests.post(
        f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()

    max_steps = obs["max_steps"]
    conversation = []
    final_score = 0.001
    steps_taken = 0
    prev_score = 0.0
    stall_count = 0
    last_action = None

    for step_num in range(max_steps):
        obs_text = json.dumps(obs, indent=2)

        # Build user message with extra context on stalls
        user_content = f"Current observation:\n{obs_text}"
        if stall_count >= 2:
            user_content += (
                "\n\nWARNING: Your last actions did not improve the score. "
                "Try a completely different action on a column you have not touched yet."
            )

        conversation.append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
        raw = call_llm(client, messages)
        if not raw:
            continue

        conversation.append({"role": "assistant", "content": raw})
        action_dict = parse_action(raw)
        if not action_dict:
            print(f"WARN: Could not parse action from: {raw[:80]}", flush=True)
            continue

        action_type = action_dict.get("action_type")

        # Guard: don't submit if obs still has issues
        if action_type == "submit" and obs_has_issues(obs):
            print("WARN: Agent tried to submit with issues remaining — blocking submit.", flush=True)
            conversation.append({
                "role": "user",
                "content": "DO NOT submit yet. The observation still shows issues. Fix them first."
            })
            continue

        # Guard: don't repeat same action twice in a row
        if action_dict == last_action:
            print("WARN: Repeated action blocked — forcing different approach.", flush=True)
            conversation.append({
                "role": "user",
                "content": "That action was already tried. Choose a DIFFERENT action on a different column."
            })
            continue

        steps_taken = step_num + 1
        print(f"[STEP] step={steps_taken} action={action_type}", flush=True)

        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
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
        raw_score = float(reward["value"])
        final_score = max(0.001, min(0.999, raw_score))

        print(f"[STEP] step={steps_taken} reward={final_score}", flush=True)

        # Track stalls
        if final_score <= prev_score:
            stall_count += 1
        else:
            stall_count = 0
        prev_score = final_score
        last_action = action_dict

        if done:
            break

    return max(0.001, min(0.999, final_score)), steps_taken

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
        ping = requests.get(f"{ENV_BASE_URL}/", headers=HEADERS, timeout=15)
        ping.raise_for_status()
        print(f"INFO: Environment reachable at {ENV_BASE_URL}", flush=True)
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {exc}", flush=True)
        sys.exit(1)

    scores = {}

    for task_id in [1, 2, 3]:
        print(f"[START] task={task_id}", flush=True)

        try:
            score, steps = run_agent_on_task(task_id, client)
        except Exception as exc:
            print(f"ERROR: task={task_id} crashed — {type(exc).__name__}: {exc}", flush=True)
            score, steps = 0.001, 0

        scores[task_id] = score
        print(f"[END] task={task_id} score={score} steps={steps}", flush=True)

    avg = sum(scores.values()) / len(scores)
    print(f"DONE average_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()