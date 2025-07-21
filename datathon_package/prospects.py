import pandas as pd
from typing import List, Optional
from datathon_package.utils import transpose_and_prepare_dataframe, expand_dict_column, detect_nulls_and_nans, remove_invalid_prospect_codigo, ingest_dataframe_to_postgres


def process_prospects_data(prospects_path: str) -> pd.DataFrame:
    """
    Processes a JSON prospects file and returns a cleaned DataFrame with a binary target label.

    Args:
        prospects_path (str): Path to the prospects JSON file.

    Returns:
        pd.DataFrame: Processed DataFrame with columns expanded, melted, and target labeled.
    """
    # Step 1: Read and transpose
    df = pd.read_json(prospects_path)
    df = transpose_and_prepare_dataframe(df)
    df = expand_dict_column(df, 'prospects', 'prospects_')

    # Step 2: Melt prospects
    fixed_columns = ['ID', 'titulo', 'modalidade']
    prospect_columns = [col for col in df.columns if col.startswith("prospect")]

    df_long = df.melt(
        id_vars=fixed_columns,
        value_vars=prospect_columns,
        var_name='prospect_num',
        value_name='prospect'
    )

    # Step 3: Clean null or empty
    df_long = df_long.dropna(subset=['prospect'])
    df_long = df_long[df_long['prospect'] != ""]
    df_long = df_long[fixed_columns + ['prospect']]

    # Step 4: Expand nested dict
    df = expand_dict_column(df_long, 'prospect', 'prospect_')
    df = df.sort_values(by='ID')

    df = remove_invalid_prospect_codigo(df)

    # Step 5: Drop unused fields
    cols_to_drop = [
        'ID',
        'prospect_nome',
        'prospect_data_candidatura',
        'prospect_ultima_atualizacao',
        'prospect_comentario',
        'titulo',
        'modalidade',
        'prospect_recrutador'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Step 6: Apply target classification
    aprovados = [
        'Aprovado',
        'Contratado pela Decision',
        'Contratado como Hunting',
        'Proposta Aceita',
        'Encaminhar Proposta'
    ]

    reprovados = [
        'Não Aprovado pelo Cliente',
        'Não Aprovado pelo RH',
        'Não Aprovado pelo Requisitante',
        'Recusado',
        'Desistiu',
        'Desistiu da Contratação',
        'Sem interesse nesta vaga'
    ]

    def classify_target(status: Optional[str]) -> Optional[int]:
        if status in aprovados:
            return 1
        elif status in reprovados:
            return 0
        return None

    df['target'] = df['prospect_situacao_candidado'].apply(classify_target)
    df = df.dropna(subset=['target'])

    ingest_dataframe_to_postgres(df, local=True, table_name="propects", if_exists="replace")

    return df


if __name__ == "__main__":
    # Caminho para o arquivo JSON
    prospects_path = './data/raw/prospects/prospects.json'

    # Processa os dados
    df = process_prospects_data(prospects_path)

    # Exibe resultados básicos
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    summary = detect_nulls_and_nans(df)
    print(summary)

