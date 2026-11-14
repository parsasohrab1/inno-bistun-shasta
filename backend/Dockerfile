# Bistun Paper Industries -- Integrated Quality & Energy Control (MV + APC)
# Backend image: FastAPI + OpenCV + scikit-learn + scipy. Serves the dashboard at /.
FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless needs libglib2.0-0; libgl1 kept as a safety net.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application + assets
COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY conftest.py pyproject.toml ./

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4); sys.exit(0)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
