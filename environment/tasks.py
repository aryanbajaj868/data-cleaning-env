import pandas as pd
import numpy as np
from typing import Callable, Dict, Any


# ─────────────────────────────────────────────
# Task 1 — Easy: Fix Column Headers
# ─────────────────────────────────────────────

def get_task_1_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "First Name ": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            " Last Name": ["Smith", "Jones", "Brown", "Wilson", "Davis"],
            "SALES_AMOUNT": [1200.50, 850.00, 2300.75, 640.00, 1800.25],
            "date-of-sale": [
                "2024-01-15",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
            ],
            "RegionCode": ["US-W", "US-E", "US-N", "US-S", "US-W"],
        }
    )


TASK_1: Dict[str, Any] = {
    "id": 1,
    "name": "fix_headers",
    "difficulty": "easy",
    "description": (
        "You have a sales dataset with messy column names containing spaces, dashes, "
        "and inconsistent casing. Rename all columns to clean snake_case format."
    ),
    "goal": (
        "Rename all 5 columns to: first_name, last_name, sales_amount, "
        "date_of_sale, region_code"
    ),
    "expected_columns": [
        "first_name",
        "last_name",
        "sales_amount",
        "date_of_sale",
        "region_code",
    ],
    "max_steps": 10,
    "get_data": get_task_1_data,
}


# ─────────────────────────────────────────────
# Task 2 — Medium: Handle Missing Data
# ─────────────────────────────────────────────

def get_task_2_data() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "employee_id": range(1, 11),
            "name": [
                "Alice", "Bob", None, "Dave", "Eve",
                "Frank", None, "Hannah", "Ivan", "Julia",
            ],
            "age": [25, None, 32, None, 28, 45, 31, None, 27, 33],
            "department": [
                "Engineering", "Marketing", "Engineering", None, "HR",
                None, "Marketing", "Engineering", None, "HR",
            ],
            "salary": [
                "75000", "60000", None, "80000", "55000",
                "90000", "62000", None, "70000", "58000",
            ],
        }
    )


TASK_2: Dict[str, Any] = {
    "id": 2,
    "name": "handle_missing",
    "difficulty": "medium",
    "description": (
        "You have an employee dataset with multiple data quality issues: "
        "missing names, ages, departments, and a salary column stored as strings with nulls."
    ),
    "goal": (
        "1. Fill missing 'name' values with 'Unknown'. "
        "2. Fill missing 'age' values with the median age. "
        "3. Fill missing 'department' values with 'Unknown'. "
        "4. Convert 'salary' column from string to float and fill missing salary with median."
    ),
    "max_steps": 15,
    "get_data": get_task_2_data,
}


# ─────────────────────────────────────────────
# Task 3 — Hard: Full Pipeline Repair
# ─────────────────────────────────────────────

def get_task_3_data() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        {
            "Customer ID ": [101, 102, 103, 104, 105, 106, 107, 102, 103, 108],
            "PRODUCT-NAME": [
                "Widget A", "Widget B", "Widget C", "Widget D", "Widget E",
                "Widget F", "Widget G", "Widget B", "Widget C", "Widget H",
            ],
            "purchase amount": [
                150.0, 200.0, None, 9999.0, 175.0,
                220.0, None, 200.0, None, 190.0,
            ],
            "PurchaseDate": [
                "2024-01", "2024-02", "2024-03", "2024-04", "2024-05",
                "2024-06", "2024-07", "2024-02", "2024-03", "2024-08",
            ],
            "CustomerAge": [25, 30, None, 28, 300, 35, 22, 30, None, 29],
        }
    )


TASK_3: Dict[str, Any] = {
    "id": 3,
    "name": "full_pipeline_repair",
    "difficulty": "hard",
    "description": (
        "You have a messy customer transactions dataset with MULTIPLE issues: "
        "messy column names, duplicate rows (same customer + product), "
        "missing values, extreme outliers in purchase_amount and customer_age."
    ),
    "goal": (
        "1. Rename all columns to snake_case: customer_id, product_name, "
        "purchase_amount, purchase_date, customer_age. "
        "2. Drop duplicate rows (same customer_id + product_name). "
        "3. Fill missing purchase_amount with median. "
        "4. Remove/fix outlier purchase_amount (value 9999 is erroneous — cap or replace). "
        "5. Fill missing customer_age with median and remove outlier age 300."
    ),
    "max_steps": 20,
    "get_data": get_task_3_data,
}


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

TASKS: Dict[int, Dict[str, Any]] = {
    1: TASK_1,
    2: TASK_2,
    3: TASK_3,
}
