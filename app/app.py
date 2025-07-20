from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
import os
import uuid
import json
from artifacts_datathon.applicants import process_applicants_data

app = Flask(__name__)

# Carrega o modelo
with open("/Users/leonardo/codigos/codigo/app/models/lgbm_oversample_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        content = request.get_json()
        print(content)

        if not content or not isinstance(content, dict):
            return jsonify({"error": "Formato de JSON inválido"}), 400

        # Salva os dados recebidos como um JSON temporário
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/temp_applicants_{temp_id}.json"

        with open(temp_path, "w") as f:
            json.dump(content, f)

        # Processa o arquivo com a função existente
        df = pd.read_json(temp_path)
        print(df.head())
        df_features = process_applicants_data(temp_path, predict=True)

        # Remove o arquivo temporário após o uso
        os.remove(temp_path)

        if df_features.empty:
            return jsonify({"error": "Nenhum dado processado"}), 400

        # Faz as predições
        preds = model.predict(df_features)
        probas = model.predict_proba(df_features)[:, 1]

        # Monta a resposta
        results = []
        for i in range(len(preds)):
            results.append({
                "applicant_id": df_features.index[i] if df_features.index.name else i,
                "prediction": int(preds[i]),
                "probability": round(float(probas[i]), 4)
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=True)
