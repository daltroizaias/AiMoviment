
# Use a imagem oficial do Python 3.12.8 como base
FROM python:3.12.8-slim-bullseye

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos do diretório local para o diretório de trabalho no contêiner
COPY . .

# Instala as dependências necessárias
RUN pip install poetry

RUN poetry config installer.max-workers 10
RUN poetry install --without doc,dev --no-interaction --no-ansi
# Expor a porta em que o Gunicorn estará executando
EXPOSE 5000

# Comando para iniciar o servidor Gunicorn
CMD ["poetry", "run","gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
