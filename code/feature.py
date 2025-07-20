import pandas as pd
from typing import Any, Dict
from pathlib import Path
import re

def generate_flags_and_category_column(
    df: pd.DataFrame,
    source_column: str,
    mapping: Dict[str, str],
    default_flag_column: str = None
) -> pd.DataFrame:
    """
    Generates binary flag columns from a categorical text column.
    Optionally adds a default flag column for unmatched values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        source_column (str): Name of the source column with text categories.
        mapping (Dict[str, str]): Dictionary where keys are source values and
                                  values are new binary column names.
        default_flag_column (str, optional): Name of the binary column for unmatched values.
                                             If None, unmatched values are ignored.

    Returns:
        pd.DataFrame: Updated DataFrame with new binary columns and the source column removed.
    """
    df = df.copy()

    # Create binary flags for mapped values
    for value, col_name in mapping.items():
        df[col_name] = df[source_column].apply(lambda x: 1 if x == value else 0)

    # Add default column for unmatched values
    if default_flag_column:
        df[default_flag_column] = df[source_column].apply(lambda x: 0 if x in mapping else 1)

    # Drop the original source column
    df.drop(columns=[source_column], inplace=True)

    return df


def process_certification_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Processes a column with comma-separated certification strings and generates:
      - A count of certifications
      - Binary range flags: certificacoes_0, certificacoes_1_3, certificacoes_5_mais

    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of the certification column (comma-separated string or empty)

    Returns:
        pd.DataFrame: Updated DataFrame with count and binary flags, original column dropped
    """
    df = df.copy()

    def count_certifications(val: Any) -> int:
        if pd.isna(val) or str(val).strip() == '':
            return 0
        return len([x for x in str(val).split(',') if x.strip() != ''])

    # Count certifications
    df['certificacoes_count'] = df[column].apply(count_certifications)

    # Create binary flags by range
    df['certificacoes_0'] = df['certificacoes_count'].apply(lambda x: 1 if x == 0 else 0)
    df['certificacoes_1_3'] = df['certificacoes_count'].apply(lambda x: 1 if 1 <= x <= 3 else 0)
    df['certificacoes_5_mais'] = df['certificacoes_count'].apply(lambda x: 1 if x >= 5 else 0)

    # Drop the original column
    df.drop(columns=[column], inplace=True)

    return df


def process_salary_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Processes a salary column with mixed formats (e.g., 'R$ 2500,00 mensal', 'R$115 p/h') and:
      - Extracts numeric value using regex
      - Classifies as horista or mensalista
      - Generates binary columns for predefined salary ranges
      - Sets missing or invalid values to 0

    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of the salary column

    Returns:
        pd.DataFrame: DataFrame with salary ranges as binary columns and original column removed
    """
    df = df.copy()

    def extract_salary(value: Any) -> float:
        if pd.isna(value) or str(value).strip() == '' or str(value).strip() == '0':
            return 0.0
        # Regex to extract numeric part: handles "R$ 2.500,00", "5000", "22000 mensais", etc.
        match = re.search(r'(\d{1,3}(?:[\.\d{3}]*)(?:,\d{2})?|\d+)', str(value))
        if match:
            raw = match.group(1).replace('.', '').replace(',', '.')
            try:
                return float(raw)
            except ValueError:
                return 0.0
        return 0.0

    df['remuneracao_valor'] = df[column].apply(extract_salary)

    # Faixas horista (valor < 1000)
    df['horista0_100'] = df['remuneracao_valor'].apply(lambda x: 1 if 0 <= x < 100 else 0)
    df['horista_100_300'] = df['remuneracao_valor'].apply(lambda x: 1 if 100 <= x < 300 else 0)
    df['horista300_500'] = df['remuneracao_valor'].apply(lambda x: 1 if 300 <= x < 500 else 0)
    df['horista500_1000'] = df['remuneracao_valor'].apply(lambda x: 1 if 500 <= x < 1000 else 0)

    # Faixas mensalista (valor >= 1000)
    df['mensalista1000_5000'] = df['remuneracao_valor'].apply(lambda x: 1 if 1000 <= x < 5000 else 0)
    df['mensalista_5000_10000'] = df['remuneracao_valor'].apply(lambda x: 1 if 5000 <= x < 10000 else 0)
    df['mensalista_10k_15k'] = df['remuneracao_valor'].apply(lambda x: 1 if 10000 <= x < 15000 else 0)
    df['mensalista20k_mais'] = df['remuneracao_valor'].apply(lambda x: 1 if x >= 20000 else 0)

    # Remove coluna original e auxiliar
    df.drop(columns=[column, 'remuneracao_valor'], inplace=True)

    return df

from datetime import datetime

def process_promotion_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Processes a date column representing the last promotion and generates binary columns
    indicating the time since that promotion in years.
    Empty or null values are treated as 5+ years (promocao_mais_5a = 1).

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Name of the promotion date column (expected format: 'dd-mm-yyyy').

    Returns:
        pd.DataFrame: DataFrame with binary range columns and original column removed.
    """
    df = df.copy()
    today = datetime.today()

    def calculate_years(date_str: Any) -> float:
        try:
            if pd.isna(date_str) or str(date_str).strip() == '':
                return 5.01  # Considered as "more than 5 years"
            date_obj = datetime.strptime(str(date_str).strip(), '%d-%m-%Y')
            delta = today - date_obj
            return round(delta.days / 365.25, 2)
        except Exception:
            return 5.01  # Fallback for parsing errors

    # Cálculo da diferença em anos
    df['anos_desde_promocao'] = df[column].apply(calculate_years)

    # Faixas em anos
    df['promocao_menos_1a'] = df['anos_desde_promocao'].apply(lambda x: 1 if x < 1 else 0)
    df['promocao_1a_2a'] = df['anos_desde_promocao'].apply(lambda x: 1 if 1 <= x < 2 else 0)
    df['promocao_2a_3a'] = df['anos_desde_promocao'].apply(lambda x: 1 if 2 <= x < 3 else 0)
    df['promocao_mais_5a'] = df['anos_desde_promocao'].apply(lambda x: 1 if x >= 5 else 0)

    # Remove coluna original e auxiliar
    df.drop(columns=[column, 'anos_desde_promocao'], inplace=True)

    return df
