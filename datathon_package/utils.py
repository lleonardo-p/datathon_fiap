import pandas as pd
from typing import Any, Dict
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os



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


def drop_constant_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica e remove colunas que contenham apenas 0s ou apenas 1s.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: DataFrame sem colunas binárias constantes.
    """
    constant_cols: List[str] = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1 and df[col].isin([0, 1]).all()
    ]

    if constant_cols:
        print(f"📌 {len(constant_cols)} colunas com apenas 0 ou 1 serão removidas:")
        for col in constant_cols:
            print(f" - {col}")
    else:
        print("✅ Nenhuma coluna binária constante encontrada.")

    return df.drop(columns=constant_cols)


import pandas as pd


def ingest_dataframe_to_postgres(df: pd.DataFrame, local, table_name: str, if_exists: str = "replace"):
    """
    Insere um DataFrame em uma tabela PostgreSQL.

    Parâmetros:
        df (pd.DataFrame): o DataFrame a ser ingerido
        table_name (str): nome da tabela de destino no PostgreSQL
        if_exists (str): comportamento se a tabela já existir:
                         - 'fail': lança erro
                         - 'replace': substitui a tabela
                         - 'append': insere os dados sem apagar a tabela
    """
    try:
        # Carrega variáveis de ambiente, se existir um .env
        load_dotenv()
        # Coleta configs do .env (ou usa valores default)
        DB_USER = os.getenv("POSTGRES_USER", "leonardo")
        DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "123456")
        DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
        DB_PORT = os.getenv("POSTGRES_PORT", "5432")
        DB_NAME = os.getenv("POSTGRES_DB", "meubanco")

        if local:
            DB_HOST = "localhost"

        # Cria a string de conexão
        DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(DATABASE_URL)

        # Envia o DataFrame
        df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False, method="multi")

        print(f"Tabela '{table_name}' atualizada com sucesso ({len(df)} registros).")

    except Exception as e:
        print("Erro ao inserir dados no PostgreSQL:")
        print(e)
