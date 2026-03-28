import pandas as pd
import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# Task 1 Grader — Fix Headers
# ─────────────────────────────────────────────

def grade_task_1(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Checks whether all columns have been renamed to the expected snake_case names.
    Partial credit: each correctly renamed column contributes 0.2 to the score.
    """
    expected = ["first_name", "last_name", "sales_amount", "date_of_sale", "region_code"]
    actual = list(df.columns)

    # Set-based check (unordered)
    set_correct = len(set(expected) & set(actual))
    set_score = set_correct / len(expected)

    # Order-based check (bonus)
    if len(actual) == len(expected):
        order_correct = sum(1 for e, a in zip(expected, actual) if e == a)
        order_score = order_correct / len(expected)
    else:
        order_score = 0.0

    # Weighted: set match matters more than order
    score = round(set_score * 0.7 + order_score * 0.3, 3)
    breakdown = {
        "columns_present": round(set_score, 3),
        "columns_ordered": round(order_score, 3),
    }
    return score, breakdown


# ─────────────────────────────────────────────
# Task 2 Grader — Handle Missing Data
# ─────────────────────────────────────────────

def grade_task_2(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Five equally-weighted criteria (0.2 each):
      1. No missing 'name' values
      2. No missing 'age' values
      3. No missing 'department' values
      4. 'salary' is numeric (float/int)
      5. No missing 'salary' values
    """
    breakdown: Dict[str, float] = {}
    scores = []

    def no_missing(col: str) -> float:
        if col not in df.columns:
            return 0.0
        missing = int(df[col].isna().sum())
        return round(1.0 if missing == 0 else max(0.0, 1.0 - missing / len(df)), 3)

    breakdown["name_no_missing"] = no_missing("name")
    scores.append(breakdown["name_no_missing"])

    breakdown["age_no_missing"] = no_missing("age")
    scores.append(breakdown["age_no_missing"])

    breakdown["department_no_missing"] = no_missing("department")
    scores.append(breakdown["department_no_missing"])

    if "salary" in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df["salary"])
        breakdown["salary_is_numeric"] = 1.0 if is_numeric else 0.0
        scores.append(breakdown["salary_is_numeric"])
        if is_numeric:
            breakdown["salary_no_missing"] = no_missing("salary")
        else:
            breakdown["salary_no_missing"] = 0.0
        scores.append(breakdown["salary_no_missing"])
    else:
        breakdown["salary_is_numeric"] = 0.0
        breakdown["salary_no_missing"] = 0.0
        scores.extend([0.0, 0.0])

    score = round(sum(scores) / len(scores), 3)
    return score, breakdown


# ─────────────────────────────────────────────
# Task 3 Grader — Full Pipeline Repair
# ─────────────────────────────────────────────

def grade_task_3(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Five equally-weighted criteria (0.2 each):
      1. All columns renamed to snake_case
      2. No duplicate (customer_id, product_name) pairs
      3. No missing purchase_amount
      4. No outlier in purchase_amount (max <= 1000)
      5. No outlier in customer_age (max <= 120) and no missing age
    """
    breakdown: Dict[str, float] = {}
    weighted: list = []

    # 1. Column names (0.20)
    expected_cols = {
        "customer_id", "product_name", "purchase_amount", "purchase_date", "customer_age"
    }
    present = len(expected_cols & set(df.columns))
    col_score = round(present / len(expected_cols), 3)
    breakdown["column_names"] = col_score
    weighted.append(col_score * 0.20)

    # 2. No duplicates (0.20)
    if "customer_id" in df.columns and "product_name" in df.columns:
        dup_count = int(df.duplicated(subset=["customer_id", "product_name"]).sum())
        dup_score = round(1.0 if dup_count == 0 else max(0.0, 1.0 - dup_count / len(df)), 3)
    else:
        dup_score = 0.0
    breakdown["no_duplicates"] = dup_score
    weighted.append(dup_score * 0.20)

    # 3. No missing purchase_amount (0.20)
    if "purchase_amount" in df.columns:
        missing = int(df["purchase_amount"].isna().sum())
        miss_score = round(1.0 if missing == 0 else max(0.0, 1.0 - missing / len(df)), 3)
    else:
        miss_score = 0.0
    breakdown["purchase_amount_no_missing"] = miss_score
    weighted.append(miss_score * 0.20)

    # 4. No purchase_amount outlier (0.20)
    if "purchase_amount" in df.columns and pd.api.types.is_numeric_dtype(df["purchase_amount"]):
        max_val = df["purchase_amount"].dropna().max()
        outlier_score = 1.0 if float(max_val) <= 1000.0 else 0.0
    else:
        outlier_score = 0.0
    breakdown["no_purchase_outlier"] = outlier_score
    weighted.append(outlier_score * 0.20)

    # 5. No age outlier + no missing age (0.20)
    if "customer_age" in df.columns and pd.api.types.is_numeric_dtype(df["customer_age"]):
        max_age = df["customer_age"].dropna().max()
        missing_age = int(df["customer_age"].isna().sum())
        age_score = round(
            (1.0 if float(max_age) <= 120.0 else 0.0) * 0.5
            + (1.0 if missing_age == 0 else 0.0) * 0.5,
            3,
        )
    else:
        age_score = 0.0
    breakdown["age_clean"] = age_score
    weighted.append(age_score * 0.20)

    total = round(sum(weighted), 3)
    return total, breakdown


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
}
