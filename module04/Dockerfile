FROM python:3.12.4-slim

COPY [ "cacert.pem", "/usr/local/lib/python3.12/site-packages/certifi/cacert.pem" ]

RUN pip install -U pip && pip install pipenv

WORKDIR /app

COPY [ "model.bin", "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "batch.py", "batch.py" ]

ENTRYPOINT [ "python", "batch.py" ]
