import pandas as pd
from typing import Dict
from utils import transpose_and_prepare_dataframe, expand_dict_column, detect_nulls_and_nans
from feature import (
    generate_flags_and_category_column,
    process_certification_column,
    process_salary_column,
    process_promotion_date_column
)


def process_applicants_data(applicants_path: str) -> pd.DataFrame:
    """
    Processes the applicants JSON file and returns a transformed DataFrame with engineered features.

    Args:
        applicants_path (str): Path to the applicants JSON file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_json(applicants_path)

    df = transpose_and_prepare_dataframe(df)

    df = expand_dict_column(df, 'infos_basicas', 'infos_basicas_')
    df = expand_dict_column(df, 'informacoes_pessoais', 'informacoes_pessoais_')
    df = expand_dict_column(df, 'informacoes_profissionais', 'informacoes_profissionais_')
    df = expand_dict_column(df, 'formacao_e_idiomas', 'formacao_e_idiomas_')
    df = expand_dict_column(df, 'cargo_atual', 'cargo_atual_')

    # Seleção de colunas desejadas
    selected_columns = [
        'ID',
        'infos_basicas_sabendo_de_nos_por',
        'infos_basicas_codigo_profissional',
        'informacoes_pessoais_estado_civil',
        'informacoes_profissionais_certificacoes',
        'informacoes_profissionais_remuneracao',
        'informacoes_profissionais_nivel_profissional',
        'formacao_e_idiomas_nivel_academico',
        'cargo_atual_data_ultima_promocao',
    ]
    df = df[selected_columns]

    # Feature: Indicação
    mapping_indicacao = {
        'Indicação de colaborador': 'ind_colaborador',
        'Indicação de cliente': 'ind_cliente'
    }

    df = generate_flags_and_category_column(
        df,
        source_column='infos_basicas_sabendo_de_nos_por',
        mapping=mapping_indicacao,
        default_flag_column='ind_outros'
    )

    # Feature: Estado Civil
    estado_civil_values = ['', 'Casado', 'Divorciado', 'Solteiro', 'Separado Judicialmente', 'União Estável', 'Viúvo']
    mapping_estado_civil = {
        value: f"estado_civil_{value.lower().replace(' ', '_') or 'vazio'}"
        for value in estado_civil_values
    }

    df = generate_flags_and_category_column(
        df,
        source_column='informacoes_pessoais_estado_civil',
        mapping=mapping_estado_civil,
        default_flag_column=None
    )

    # Feature: Nível Acadêmico
    df['formacao_e_idiomas_nivel_academico'] = df['formacao_e_idiomas_nivel_academico'].fillna('').replace('', 'nivel_academico_vazio')

    nivel_academico_valores = [
        'nivel_academico_vazio', 'Pós Graduação Completo', 'Ensino Superior Completo', 'Mestrado Completo',
        'Ensino Médio Completo', 'Ensino Técnico Completo', 'Ensino Superior Incompleto',
        'Ensino Superior Cursando', 'Pós Graduação Incompleto', 'Mestrado Incompleto',
        'Pós Graduação Cursando', 'Mestrado Cursando', 'Ensino Técnico Cursando',
        'Doutorado Incompleto', 'Ensino Médio Incompleto', 'Ensino Fundamental Completo',
        'Doutorado Completo', 'Ensino Técnico Incompleto', 'Ensino Médio Cursando',
        'Ensino Fundamental Incompleto', 'Doutorado Cursando', 'Ensino Fundamental Cursando'
    ]

    mapping_nivel_academico = {
        valor: f"nivel_academico_{valor.lower().replace(' ', '_').replace('ç', 'c').replace('ã', 'a')}"
        for valor in nivel_academico_valores
    }

    df = generate_flags_and_category_column(
        df,
        source_column='formacao_e_idiomas_nivel_academico',
        mapping=mapping_nivel_academico
    )

    # Feature: Nível Profissional
    df['informacoes_profissionais_nivel_profissional'] = (
        df['informacoes_profissionais_nivel_profissional']
        .fillna('')
        .replace(
            to_replace=['', 'Outro', 'outro', 'NA', 'nA', 'na', 'Na', 'None'],
            value='outros'
        )
    )

    nivel_profissional_valores = df['informacoes_profissionais_nivel_profissional'].unique()

    mapping_nivel_profissional: Dict[str, str] = {
        valor: f"nivel_profissional_{valor.lower().replace(' ', '_').replace('ç', 'c')}"
        for valor in nivel_profissional_valores
    }

    df = generate_flags_and_category_column(
        df,
        source_column='informacoes_profissionais_nivel_profissional',
        mapping=mapping_nivel_profissional
    )

    # Feature: Certificações
    df = process_certification_column(df, 'informacoes_profissionais_certificacoes')

    # Feature: Salário
    df = process_salary_column(df, 'informacoes_profissionais_remuneracao')

    # Feature: Promoção
    df = process_promotion_date_column(df, 'cargo_atual_data_ultima_promocao')

    return df


if __name__ == "__main__":
    applicants_path = './data/applicants/applicants.json'
    df = process_applicants_data(applicants_path)

    print("Colunas geradas:")
    print(df.columns.tolist())
    print("\nShape final:", df.shape)
    summary = detect_nulls_and_nans(df)
    print(summary)
