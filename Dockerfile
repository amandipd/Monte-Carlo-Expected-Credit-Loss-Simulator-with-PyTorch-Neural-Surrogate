FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-root --only main

COPY src/ src/
RUN mkdir -p /app/results /app/models /app/data

CMD ["python", "src/redis/consumer.py"]