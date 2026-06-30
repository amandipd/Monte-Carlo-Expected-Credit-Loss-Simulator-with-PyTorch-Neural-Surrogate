FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
COPY src/ src/
RUN poetry install --no-interaction

RUN mkdir -p /app/results /app/models /app/data

CMD ["python", "-m", "risk_engine.queue.consumer"]
