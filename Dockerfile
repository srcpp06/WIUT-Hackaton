FROM python:3.12-slim
# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch wheels (the server has no GPU)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY src/ src/
COPY templates/ templates/
COPY static/ static/
# The demo bank is generated, not shipped: same seed, same data (CC0)
RUN python -m src.generator --out data/demo --accounts 2000 --seed 7
RUN useradd -u 1000 -m app && mkdir -p var && chown app var
USER app
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
