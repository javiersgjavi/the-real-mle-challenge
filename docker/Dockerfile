FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

ENV PYTHONUNBUFFERED=1
