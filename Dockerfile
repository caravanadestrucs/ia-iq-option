FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# dependencias del sistema necesarias para algunos wheels (xgboost, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5001

CMD ["python", "server.py"]
