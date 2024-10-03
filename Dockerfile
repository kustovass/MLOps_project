FROM python:3.10-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev
