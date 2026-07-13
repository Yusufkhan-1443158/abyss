FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install --legacy-peer-deps && npm ls ajv || npm install ajv@8.12.0 --legacy-peer-deps
COPY frontend/ ./
ARG REACT_APP_CLIENT_TIER
ENV REACT_APP_CLIENT_TIER=$REACT_APP_CLIENT_TIER
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgdal-dev gdal-bin libgeos-dev libproj-dev \
    libspatialindex-dev libhdf5-dev libnetcdf-dev \
    wget gnupg ca-certificates \
    # Chromium deps for Playwright headless screenshots
    chromium fonts-liberation libnss3 libatk-bridge2.0-0 libdrm2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2 \
    libpango-1.0-0 libcairo2 libxshmfence1 libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (smaller, no CUDA bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Install Playwright and point to system Chromium
RUN pip install --no-cache-dir playwright && \
    PLAYWRIGHT_BROWSERS_PATH=/usr/lib playwright install chromium 2>/dev/null || true
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/lib
ENV CHROMIUM_PATH=/usr/bin/chromium

COPY backend/ ./backend/
# In-situ calibration surveys (KP / KP_EMAL / OMC xyz + SWOT_Dhanna shp).
# Without them _load_xyz_full silently finds 0 points and every run
# calibrates on GEBCO anchors only — depths come out flattened/garbage.
COPY validation/ ./validation/
COPY --from=frontend-build /app/frontend/build ./frontend/build
# Full user guide served by GET /api/user-guide (Help panel links to it).
COPY USER_GUIDE.md ./USER_GUIDE.md
# Prerendered sample SDB products (~1 MB) the persistent Results store
# and the Khalifa-Port VHR button hydrate from on a fresh container.
COPY samples/ ./samples/
RUN mkdir -p /tmp/bathy/cache /tmp/bathy/exports /tmp/bathy/screenshots
ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "600", "--preload", "backend.app:app"]
