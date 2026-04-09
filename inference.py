"""
DataCleaningEnv — Optimised Inference Script
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
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")   # upgraded from gpt-4o-mini
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

    print(f"INFO: LLM proxy base_url={LLM_BASE_URL} model={MODEL_NAME}", flush=True)

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
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world-class data cleaning agent. You are given an observation
describing a dataset's current state and must choose the single best next action.

━━━ AVAILABLE ACTIONS ━━━
{"action_type": "rename_all_snake_case", "parameters": {}}
{"action_type": "rename_column", "parameters": {"old_name": "...", "new_name": "..."}}
{"action_type": "convert_type", "parameters": {"column": "...", "dtype": "float|int|str"}}
{"action_type": "fill_missing", "parameters": {"column": "...", "strategy": "mean|median|mode|value", "value": <only if strategy=value>}}
{"action_type": "drop_duplicates", "parameters": {"subset": ["col1", "col2", ...]}}
{"action_type": "remove_outliers", "parameters": {"column": "...", "method": "iqr|cap", "fill": "median|mean"}}
{"action_type": "submit", "parameters": {}}

━━━ STRICT EXECUTION ORDER ━━━
PHASE 1 — RENAME
  • Always call rename_all_snake_case first, exactly once.

PHASE 2 — TYPE CONVERSION
  • For every column that contains numeric data stored as strings (e.g. "$1,200", "3.5%", "1,000"):
    call convert_type with dtype=float or dtype=int.
  • Do this for ALL such columns before moving on.

PHASE 3 — FILL MISSING VALUES
  • For every column with missing values:
    - Numeric column → strategy=median
    - Categorical/text column → strategy=mode
  • Handle ALL columns with missing values before moving on.

PHASE 4 — DROP DUPLICATES
  • Call drop_duplicates once with all column names as subset.

PHASE 5 — REMOVE OUTLIERS
  • For every numeric column: call remove_outliers with method=iqr, fill=median.
  • Handle ALL numeric columns before moving on.

PHASE 6 — SUBMIT
  • Only call submit when ALL of the following are true:
    - Zero missing values in every column
    - Zero duplicate rows
    - All numeric columns have had outliers removed
  • If any issue remains, go back and fix it first.

━━━ RULES ━━━
• Respond with ONLY a single JSON object — no explanation, no markdown, no extra text.
• Never repeat the exact same action consecutively.
• Never skip a phase.
• Never submit early.
• Read the column names from the observation exactly — do not guess or invent column names.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis prompt — two-pass approach
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSIS_PROMPT = """You are a data cleaning planner. Given the observation below,
produce a complete ordered cleaning plan as a JSON array of actions to take.

Return ONLY a JSON array like:
[
  {"action_type": "rename_all_snake_case", "parameters": {}},
  {"action_type": "convert_type", "parameters": {"column": "salary", "dtype": "float"}},
  ...
  {"action_type": "submit", "parameters": {}}
]

Rules:
- Start with rename_all_snake_case
- Convert all string-encoded numeric columns to float/int
- Fill all missing values (median for numeric, mode for categorical)
- Drop duplicates
- Remove outliers from all numeric columns (iqr method, median fill)
- End with submit
- Use EXACT column names from the observation
- Return ONLY valid JSON array, nothing else
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, messages: list, max_tokens: int = 500) -> str:
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"WARN: LLM attempt {attempt+1} failed: {exc}", flush=True)
    return ""


def parse_action(raw: str) -> dict | None:
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json\n"):
                clean = clean[5:]
        clean = clean.strip()
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        return None


def parse_plan(raw: str) -> list:
    """Parse the diagnosis plan — a JSON array of actions."""
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json\n"):
                clean = clean[5:]
        clean = clean.strip()
        result = json.loads(clean)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


def obs_has_issues(obs: dict) -> bool:
    try:
        issues = obs.get("issues", {})
        if isinstance(issues, dict) and any(v for v in issues.values()):
            return True
        stats = obs.get("stats", {})
        missing = stats.get("missing_values", {})
        if any(v > 0 for v in missing.values()):
            return True
        if stats.get("duplicate_rows", 0) > 0:
            return True
        return False
    except Exception:
        return True


def get_numeric_columns(obs: dict) -> list:
    try:
        dtypes = obs.get("stats", {}).get("dtypes", {})
        return [col for col, dtype in dtypes.items()
                if dtype in ("float64", "int64", "float32", "int32", "float", "int")]
    except Exception:
        return []


def get_missing_columns(obs: dict) -> list:
    try:
        missing = obs.get("stats", {}).get("missing_values", {})
        return [col for col, count in missing.items() if count > 0]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Two-pass agent: diagnose first, then execute
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(client: OpenAI, obs: dict) -> list:
    """Ask the LLM to produce a full cleaning plan upfront."""
    obs_text = json.dumps(obs, indent=2)
    print("INFO: Running diagnosis pass...", flush=True)
    raw = call_llm(client, [
        {"role": "system", "content": DIAGNOSIS_PROMPT},
        {"role": "user", "content": f"Observation:\n{obs_text}"},
    ], max_tokens=1500)
    plan = parse_plan(raw)
    if plan:
        print(f"INFO: Plan has {len(plan)} steps: {[a.get('action_type') for a in plan]}", flush=True)
    else:
        print("WARN: Diagnosis failed, falling back to reactive mode.", flush=True)
    return plan


def run_agent_on_task(task_id: int, client: OpenAI):
    resp = requests.post(
        f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()

    print(f"INFO: task={task_id} obs_keys={list(obs.keys())}", flush=True)
    print(f"INFO: task={task_id} stats={json.dumps(obs.get('stats', {}), indent=2)}", flush=True)

    max_steps = obs["max_steps"]
    final_score = 0.001
    steps_taken = 0
    prev_score = 0.0
    stall_count = 0
    last_action = None

    # ── Pass 1: Diagnosis ──
    plan = diagnose(client, obs)
    plan_index = 0

    # ── Pass 2: Execute ──
    conversation = []

    for step_num in range(max_steps):

        # Try plan-based action first
        action_dict = None
        if plan and plan_index < len(plan):
            candidate = plan[plan_index]
            if candidate != last_action:
                action_dict = candidate
                plan_index += 1
                print(f"INFO: Using planned action {plan_index}/{len(plan)}", flush=True)
            else:
                plan_index += 1  # skip duplicate

        # Fall back to reactive LLM if plan is exhausted or repeated
        if not action_dict:
            obs_text = json.dumps(obs, indent=2)
            user_content = f"Current observation:\n{obs_text}"
            if stall_count >= 2:
                user_content += (
                    "\n\nWARNING: Score is not improving. "
                    "Try a completely different action on a column not yet cleaned."
                )
            conversation.append({"role": "user", "content": user_content})
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
            raw = call_llm(client, messages)
            if not raw:
                continue
            conversation.append({"role": "assistant", "content": raw})
            action_dict = parse_action(raw)
            if not action_dict:
                print(f"WARN: Could not parse: {raw[:80]}", flush=True)
                continue

        action_type = action_dict.get("action_type")

        # Guard: block premature submit
        if action_type == "submit" and obs_has_issues(obs):
            print("WARN: Blocking premature submit — issues remain.", flush=True)
            if plan:
                # inject fix steps into plan
                missing_cols = get_missing_columns(obs)
                numeric_cols = get_numeric_columns(obs)
                extra = []
                for col in missing_cols:
                    extra.append({"action_type": "fill_missing",
                                  "parameters": {"column": col, "strategy": "median"}})
                for col in numeric_cols:
                    extra.append({"action_type": "remove_outliers",
                                  "parameters": {"column": col, "method": "iqr", "fill": "median"}})
                extra.append({"action_type": "submit", "parameters": {}})
                plan = plan[:plan_index] + extra
            conversation.append({
                "role": "user",
                "content": "DO NOT submit yet — issues remain. Fix all missing values and outliers first."
            })
            continue

        # Guard: block exact repeat
        if action_dict == last_action:
            print("WARN: Blocking repeated action.", flush=True)
            plan_index += 1
            continue

        steps_taken = step_num + 1
        print(f"[STEP] step={steps_taken} action={action_type} params={action_dict.get('parameters', {})}", flush=True)

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