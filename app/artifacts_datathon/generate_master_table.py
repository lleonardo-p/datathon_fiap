import pandas as pd
from artifacts_datathon.prospects import process_prospects_data
from artifacts_datathon.applicants import process_applicants_data
from artifacts_datathon.utils import drop_constant_binary_columns, ingest_dataframe_to_postgres


def generate_master_table(
    applicants_path: str,
    prospects_path: str,
    output_path: str = './data/processed/master_table.parquet'
) -> pd.DataFrame:
    """
    Generates and saves a master table by merging processed applicants and prospects data.

    Args:
        applicants_path (str): Path to the applicants JSON file.
        prospects_path (str): Path to the prospects JSON file.
        output_path (str): Path to save the output parquet file.

    Returns:
        pd.DataFrame: Merged master table with only matched records.
    """
    df_applicants = process_applicants_data(applicants_path)
    df_prospects = process_prospects_data(prospects_path)

    # Converter IDs para inteiro, ignorando erros
    df_applicants['ID'] = pd.to_numeric(df_applicants['ID'], errors='coerce').astype('Int64')
    df_prospects['prospect_codigo'] = pd.to_numeric(df_prospects['prospect_codigo'], errors='coerce').astype('Int64')

    # Remover registros com IDs inválidos
    df_applicants = df_applicants.dropna(subset=['ID'])
    df_prospects = df_prospects.dropna(subset=['prospect_codigo'])

    # Fazer merge
    df_master = pd.merge(
        df_applicants,
        df_prospects,
        left_on='ID',
        right_on='prospect_codigo',
        how='inner'
    )

    df_master = df_master.drop(columns=['prospect_situacao_candidado'], errors='ignore')
    df_master = drop_constant_binary_columns(df_master)

    # Salvar em Parquet
    df_master.to_parquet(output_path, index=False)

    ingest_dataframe_to_postgres(df_master, table_name="master_table", if_exists="replace")


    return df_master


if __name__ == "__main__":
    applicants_path = './data/applicants/applicants.json'
    prospects_path = './data/prospects/prospects.json'

    df_master = generate_master_table(applicants_path, prospects_path)
    print("Master table saved. Shape:", df_master.shape)
