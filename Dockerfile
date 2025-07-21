FROM python:3.13-rc-slim

WORKDIR /app

# Instala o vim e a libgomp (necessária para o LightGBM)
RUN apt-get update && apt-get install -y vim libgomp1 && apt-get clean

# Copia código e dados
COPY ./app ./app
COPY ./data ./data
COPY requirements.txt .

# Instala dependências Python
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5007

# Ajusta os caminhos conforme estrutura de diretórios
CMD ["sh", "-c",  "python app.py"]