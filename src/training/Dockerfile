FROM python:3.12-slim

# System deps: OpenMP for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Pin setuptools to keep pkg_resources available (mlflow requires it)
RUN pip install --no-cache-dir --upgrade pip wheel "setuptools==69.5.1"

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

COPY train.py /work/train.py

CMD ["python", "train.py"]
