# Usa imagem oficial Python 3.10.13 slim
FROM python:3.10.13-slim

# Define diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (libgomp é necessária pro LightGBM)
RUN apt-get update && apt-get install -y \
    vim \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia os diretórios e arquivos necessários
COPY ./api ./api
COPY ./pipelines ./pipelines
COPY ./models ./models
COPY ./data ./data
COPY ./datathon_package ./datathon_package
COPY requirements.txt .
COPY .env .env

# Instala dependências do Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Define o PYTHONPATH para permitir imports do pacote local
ENV PYTHONPATH="/app"

# Expõe a porta que será usada pela aplicação
EXPOSE 5007

# Comando de execução padrão
CMD ["sh", "-c", "python api/app.py"]