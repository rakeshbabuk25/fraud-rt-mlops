FROM python:3.12-slim

# LightGBM needs OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Keep pkg_resources available (mlflow imports it)
RUN pip install --no-cache-dir --upgrade pip wheel "setuptools==69.5.1"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
