# Abyss API — FastAPI Backend with GDAL/Rasterio + (optional) OpenJPH (HTJ2K)
# Supports COG-first raster processing, dynamic tile rendering, and deferred
# HTJ2K encoding for the deep-zoom viewer.

FROM python:3.11-slim

# System deps: GDAL, rasterio, OpenJPEG (for legacy JP2 decode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libopenjp2-7 \
    libopenjp2-tools \
    libtiff-tools \
    libpq-dev \
    gcc \
    g++ \
    make \
    curl \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build OpenJPH from source (HTJ2K encoder/decoder). HTJ2K is OFF the instant-load
# critical path in Abyss (deferred / generated on demand for the deep-zoom viewer),
# so a failed clone/build (e.g. offline build host) must NOT break the image — the
# deferred hi-res task degrades gracefully when ojph_compress is absent.
RUN ( git clone --depth 1 https://github.com/aous72/OpenJPH.git /tmp/OpenJPH \
    && cd /tmp/OpenJPH \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/OpenJPH ) || echo "WARN: OpenJPH build skipped — HTJ2K deep-zoom disabled"

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Auth test suite — runs in-container against live PG/Redis:
#   docker compose exec fastapi pytest
COPY tests/ ./tests/
COPY pytest.ini .

# Create processing directories
RUN mkdir -p /tmp/raster-processing /tmp/raster-tiles

EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips=* so request.client.host is the real
# client behind nginx (required by the XFF-aware rate limiter — otherwise every
# request shares one bucket). nginx already sets X-Forwarded-For/-Proto.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--proxy-headers", "--forwarded-allow-ips=*"]
