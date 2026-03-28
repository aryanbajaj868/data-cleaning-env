---
title: DataCleaningEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🧹 DataCleaningEnv

An **OpenEnv-compliant** reinforcement learning environment where AI agents learn to clean real-world tabular datasets. Agents receive structured observations about messy CSV data and must apply a sequence of cleaning actions to maximise a graded reward signal.

---

## 🌍 Environment Description & Motivation

Data cleaning is one of the most time-consuming tasks in real-world data pipelines — data scientists spend [up to 80 % of their time](https://www.forbes.com/sites/gilpress/2016/03/23/data-preparation-most-time-consuming-least-enjoyable-data-science-task-survey-says/) on it. Yet there is no standard benchmark for training agents to automate this work.

**DataCleaningEnv** fills that gap. The environment:
- Presents a messy tabular dataset at each episode reset.
- Lets the agent apply structured cleaning operations step-by-step.
- Scores quality using deterministic, interpretable graders.
- Provides partial-progress rewards so the agent can learn incrementally.

---

## 📦 Project Structure

```
data-cleaning-env/
├── app.py                  # FastAPI server (OpenEnv HTTP interface)
├── inference.py            # Baseline agent script (OpenAI client)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Container definition
├── requirements.txt
├── README.md
└── environment/
    ├── __init__.py
    ├── models.py           # Pydantic models: Observation, Action, Reward
    ├── tasks.py            # Task data generators (3 tasks)
    ├── graders.py          # Deterministic graders → 0.0–1.0
    └── env.py              # DataCleaningEnv class (step / reset / state)
```

---

## 🎯 Tasks

| # | Name | Difficulty | Max Steps | Description |
|---|------|-----------|-----------|-------------|
| 1 | `fix_headers` | Easy | 10 | Rename 5 messy column headers to snake_case |
| 2 | `handle_missing` | Medium | 15 | Impute missing values; coerce salary strings to float |
| 3 | `full_pipeline_repair` | Hard | 20 | Fix column names + duplicates + missing values + outliers |

### Task 1 — Fix Headers (Easy)
The agent receives a sales dataset with column names like `"First Name "`, `" Last Name"`, `"SALES_AMOUNT"`, `"date-of-sale"`, `"RegionCode"`.  
**Goal:** rename all to `first_name`, `last_name`, `sales_amount`, `date_of_sale`, `region_code`.

### Task 2 — Handle Missing Data (Medium)
Employee dataset with missing names, ages, departments, and a salary column stored as strings.  
**Goal:** fill missing string columns with `"Unknown"`, fill numeric columns with median, convert salary to `float`.

### Task 3 — Full Pipeline Repair (Hard)
Customer transactions with messy column names, duplicate rows, missing purchase amounts, an erroneous `9999` purchase, and an impossible age of `300`.  
**Goal:** rename → deduplicate → impute → remove outliers. All five criteria must pass for a perfect score.

---

## 🔭 Observation Space

```json
{
  "task_id": 1,
  "task_description": "...",
  "goal": "...",
  "columns": [
    {
      "name": "First Name ",
      "dtype": "object",
      "missing_count": 0,
      "sample_values": ["Alice", "Bob", "Carol"]
    }
  ],
  "row_count": 5,
  "duplicate_count": 0,
  "step_count": 0,
  "max_steps": 10,
  "done": false,
  "message": "Environment reset. Start cleaning!"
}
```

---

## ⚡ Action Space

```json
{ "action_type": "rename_all_snake_case", "parameters": {} }

{ "action_type": "rename_column",
  "parameters": { "old_name": "SALES_AMOUNT", "new_name": "sales_amount" } }

{ "action_type": "fill_missing",
  "parameters": { "column": "age", "strategy": "median" } }

{ "action_type": "fill_missing",
  "parameters": { "column": "name", "strategy": "value", "value": "Unknown" } }

{ "action_type": "drop_duplicates",
  "parameters": { "subset": ["customer_id", "product_name"] } }

{ "action_type": "convert_type",
  "parameters": { "column": "salary", "dtype": "float" } }

{ "action_type": "remove_outliers",
  "parameters": { "column": "purchase_amount", "method": "iqr", "fill": "median" } }

{ "action_type": "submit", "parameters": {} }
```

---

## 🏆 Reward Function

Rewards are **non-sparse** — the agent receives a score after every step, measuring how close the current dataset is to fully clean. Invalid actions apply a small penalty (`-0.05`).

Each task has its own grader that returns a weighted sum over criteria:

- **Task 1:** `(set_match × 0.7) + (order_match × 0.3)`
- **Task 2:** average of 5 binary/partial criteria
- **Task 3:** 5 equally-weighted criteria (column names, dedup, missing, outlier × 2)

A breakdown dict is returned with every reward so the agent can see exactly which criteria it is still failing.

---

## 🚀 Setup & Usage

### Local development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# 3. Interact via HTTP
curl -X POST http://localhost:7860/reset?task_id=1
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "rename_all_snake_case", "parameters": {}}'
curl http://localhost:7860/state
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run the baseline agent

```bash
# Point to your running environment (Local or Hugging Face Space)
export ENV_URL=http://localhost:7860

# Standard Hackathon Variables for the LLM
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...  # Your OpenAI API Key goes here based on their spec

python inference.py
```

---

## 📊 Baseline Scores

Collected with `llama-3.3-70b-versatile` (temperature = 0):

| Task | Difficulty | Score | Status |
|------|-----------|-------|--------|
| 1 — fix_headers | Easy | 1.000 | ✅ Solved |
| 2 — handle_missing | Medium | 1.000 | ✅ Solved |
| 3 — full_pipeline_repair | Hard | 1.000 | ✅ Solved |
| **Average** | | **1.000** | |
---

## 🤗 HF Space Deployment

1. Create a new HF Space (Docker SDK).
2. Push this repository to the Space.
3. Set the following Space secrets:
   - `OPENAI_API_KEY`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. The Space will auto-build and expose the API at `https://huggingface.co/spaces/<username>/data-cleaning-env`.

---

## 📋 Pre-Submission Checklist

- [x] HF Space deploys and returns 200 on `GET /`
- [x] `POST /reset` returns a valid `DatasetObservation`
- [x] `openenv.yaml` present with all required fields
- [x] 3 tasks with graders returning scores in `[0.0, 1.0]`
- [x] Dockerfile builds and starts cleanly
- [x] `inference.py` in root, uses OpenAI client, reads env vars
- [x] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `OPENAI_API_KEY` all used
- [x] Baseline runs in < 20 minutes on 2 vCPU / 8 GB RAM

---
