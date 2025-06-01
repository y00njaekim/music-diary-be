ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && mkdir -p /external_libs && \
    git clone --recursive https://github.com/CPJKU/madmom.git /external_libs/madmom && \
    sed -i 's/from numpy\.math cimport INFINITY/from libc.math cimport INFINITY/g' /external_libs/madmom/madmom/ml/hmm.pyx && \
    sed -i 's/from numpy\.math cimport INFINITY/from libc.math cimport INFINITY/g' /external_libs/madmom/madmom/features/beats_crf.pyx 

COPY ./app /app

RUN pip install poetry
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

EXPOSE 5000

ENV PORT=5000

CMD ["python3", "src/main.py"]
