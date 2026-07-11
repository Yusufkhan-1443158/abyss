# Abyss Bathymetry Service — ports the real DL-Pro v3 SDB model (PyTorch).
# CPU-only torch keeps the image small (no ~2GB CUDA wheel). rasterio ships GDAL.
FROM python:3.11-slim

# curl for healthchecks; libexpat1 is required by rasterio's bundled GDAL.
RUN apt-get update && apt-get install -y --no-install-recommends curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (separate index → avoids the CUDA build).
# >=2.7: numpy-2 compatible, matches the sdb_engine training environment.
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + vendored engine + model weights.
COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
