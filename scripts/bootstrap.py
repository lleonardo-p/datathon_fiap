# app/bootstrap.py

from datathon_package.generate_master_table import generate_master_table
import os

if __name__ == "__main__":
    print("Diretório atual de execução:", os.getcwd())
    print("[BOOTSTRAP] Gerando tabela mestra...")
    applicants_path = './data/raw/applicants/applicants.json'
    prospects_path = './data/raw/prospects/prospects.json'
    df_master = generate_master_table(applicants_path, prospects_path,local=True)
    print("[BOOTSTRAP] Concluído.")
