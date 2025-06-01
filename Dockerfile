FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    mkdir -p /external_libs && \
    git clone --recursive https://github.com/CPJKU/madmom.git /external_libs/madmom && \
    sed -i 's/from numpy\.math cimport INFINITY/from libc.math cimport INFINITY/g' /external_libs/madmom/madmom/ml/hmm.pyx

COPY ./app /app
ENV PYTHONPATH /app/src

RUN pip install poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

CMD ["python3", "src/main.py"]