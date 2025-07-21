import pandas as pd
import pandas.testing as pdt
from datathon_package.utils import transpose_and_prepare_dataframe, expand_dict_column, remove_invalid_prospect_codigo, detect_nulls_and_nans, drop_constant_binary_columns

def test_transpose_and_prepare_dataframe():
    # DataFrame de entrada
    input_df = pd.DataFrame({
        'id_3': [7, 8],
        'id_1': [1, 2],
        'id_2': [4, 5]
    }, index=['feature_1', 'feature_2'])

    # DataFrame esperado após transformação
    expected_df = pd.DataFrame({
        'ID': ['id_1', 'id_2', 'id_3'],
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8]
    })

    # Chamada da função
    result_df = transpose_and_prepare_dataframe(input_df)

    # Comparação
    pdt.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_expand_dict_column():
    # DataFrame de entrada com coluna de dicionários
    input_df = pd.DataFrame({
        'id': [1, 2],
        'info': [{'a': 10, 'b': 20}, {'a': 30, 'b': 40}]
    })

    # DataFrame esperado após expansão
    expected_df = pd.DataFrame({
        'id': [1, 2],
        'info_a': [10, 30],
        'info_b': [20, 40]
    })

    # Executa a função
    result_df = expand_dict_column(input_df, column='info', prefix='info_')

    # Verifica se os DataFrames são iguais (ignorando index)
    pdt.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_remove_invalid_prospect_codigo():
    # DataFrame de entrada com valores inválidos e válidos
    input_df = pd.DataFrame({
        'prospect_codigo': ['123', '', 'nan', 'na', 'none', None, '456'],
        'nome': ['Ana', 'Bia', 'Carla', 'Dani', 'Eva', 'Fabi', 'Gustavo']
    })

    # Esperado: apenas as linhas com '123' e '456'
    expected_df = pd.DataFrame({
        'prospect_codigo': ['123', '456'],
        'nome': ['Ana', 'Gustavo']
    })

    # Executa a função
    result_df = remove_invalid_prospect_codigo(input_df)

    # Verifica igualdade ignorando índice
    pdt.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_detect_nulls_and_nans():
    # DataFrame de entrada com vários tipos de "valores inválidos"
    df = pd.DataFrame({
        'col1': [1, 2, None, 'NA', ''],
        'col2': ['n/a', 'valid', 'NaN', 'text', None],
        'col3': [10, 20, 30, 40, 50]
    })

    # DataFrame esperado com as contagens
    expected = pd.DataFrame({
        'column': ['col1', 'col2', 'col3'],
        'null_count': [1, 1, 0],
        'not_a_number_string_count': [2, 2, 0],
        'total_suspect_values': [3, 3, 0]
    }).sort_values(by='total_suspect_values', ascending=False).reset_index(drop=True)

    # Executa a função
    result = detect_nulls_and_nans(df)

    # Verifica igualdade
    pdt.assert_frame_equal(result, expected)

def test_drop_constant_binary_columns():
    # DataFrame de entrada
    df = pd.DataFrame({
        'all_zeros': [0, 0, 0],
        'all_ones': [1, 1, 1],
        'mixed_binary': [0, 1, 1],
        'non_binary': [5, 6, 7]
    })

    # Esperado: apenas colunas não constantes devem permanecer
    expected_df = pd.DataFrame({
        'mixed_binary': [0, 1, 1],
        'non_binary': [5, 6, 7]
    })

    # Executa função
    result_df = drop_constant_binary_columns(df)

    # Verifica igualdade ignorando índice
    pdt.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))
