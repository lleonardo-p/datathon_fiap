import pandas as pd
from typing import Any, Dict
from pathlib import Path



def transpose_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transposes a DataFrame assuming the index contains IDs,
    resets the index, renames it to 'ID', and sorts by ID.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: A transformed DataFrame with 'ID' as a column.
    """
    df = df.T
    df = df.reset_index()
    df = df.rename(columns={'index': 'ID'})
    df = df.sort_values(by='ID')
    return df


def expand_dict_column(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """
    Expands a dictionary-type column into multiple prefixed columns.

    Args:
        df (pd.DataFrame): The original DataFrame.
        column (str): The name of the column to expand.
        prefix (str): The prefix for the new expanded columns.

    Returns:
        pd.DataFrame: The updated DataFrame with the expanded columns.
    """
    if column in df.columns:
        try:
            expanded = pd.json_normalize(df[column]).add_prefix(prefix)
            df = df.drop(columns=[column])
            df = pd.concat([df, expanded], axis=1)
        except Exception as e:
            print(f"Error expanding column '{column}': {e}")
    return df


import pandas as pd

def remove_invalid_prospect_codigo(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (~df['prospect_codigo'].astype(str).str.strip().isin(['', 'nan', 'na', 'none']))
        & (~df['prospect_codigo'].isnull())
    ]


def detect_nulls_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and counts nulls (NaN) and string-represented 'not a number' values in all columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A summary DataFrame with counts of null and 'not a number' values per column.
    """
    not_a_number_strings = ['na', 'n/a', 'nan', 'NaN', 'NA', 'N/A', '', None]

    results = []
    for col in df.columns:
        n_nulls = df[col].isnull().sum()
        n_not_a_number_str = df[col].astype(str).str.strip().str.lower().isin(not_a_number_strings).sum()
        total = n_nulls + n_not_a_number_str
        results.append({
            'column': col,
            'null_count': n_nulls,
            'not_a_number_string_count': n_not_a_number_str,
            'total_suspect_values': total
        })

    return pd.DataFrame(results).sort_values(by='total_suspect_values', ascending=False).reset_index(drop=True)
