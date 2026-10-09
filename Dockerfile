FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRAUD_DATA_DIR=/app/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraud_app ./fraud_app
COPY index.html .
RUN useradd --create-home appuser && mkdir /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "fraud_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
