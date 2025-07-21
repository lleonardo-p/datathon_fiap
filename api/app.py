from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
import os
import uuid
import json
from datathon_package.applicants import process_applicants_data

app = Flask(__name__)

# Caminho absoluto ao diretório atual (onde este script está localizado)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho do modelo
MODEL_PATH = os.path.join("models", "lgbm_oversample_model.pkl")

# Carrega o modelo
print(f"[INFO] Carregando modelo de: {MODEL_PATH}")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[INFO] Modelo carregado com sucesso.")
except Exception as e:
    print(f"[ERRO] Falha ao carregar modelo: {e}")
    raise

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("[INFO] Requisição recebida no /predict")
        content = request.get_json()
        print("[DEBUG] JSON recebido:", content)

        if not content or not isinstance(content, dict):
            return jsonify({"error": "Formato de JSON inválido"}), 400

        # Salva os dados recebidos como um JSON temporário
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/temp_applicants_{temp_id}.json"

        with open(temp_path, "w") as f:
            json.dump(content, f)

        # Processa o arquivo com a função existente
        df = pd.read_json(temp_path)
        print("[INFO] Pré-processamento inicial:")
        print(df.head())

        df_features = process_applicants_data(temp_path, predict=True)

        os.remove(temp_path)

        if df_features.empty:
            return jsonify({"error": "Nenhum dado processado"}), 400

        # Predições
        preds = model.predict(df_features)
        probas = model.predict_proba(df_features)[:, 1]

        results = []
        for i in range(len(preds)):
            results.append({
                "applicant_id": df_features.index[i] if df_features.index.name else i,
                "prediction": int(preds[i]),
                "probability": round(float(probas[i]), 4)
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        print("[ERRO] Erro na predição:", e)
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=True)
