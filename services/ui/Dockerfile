FROM python:3.12-slim

WORKDIR /app

# Install pip tools
RUN pip install --no-cache-dir pip --upgrade

# Copy internal packages first (layer cache friendly)
COPY packages/raglab-common ./packages/raglab-common
COPY packages/raglab-chunkers ./packages/raglab-chunkers
COPY packages/raglab-retrievers ./packages/raglab-retrievers

# Copy this service
COPY services/ui ./services/ui

# Install packages then service
RUN pip install --no-cache-dir \
    ./packages/raglab-common \
    ./packages/raglab-chunkers \
    ./packages/raglab-retrievers \
    ./services/ui

EXPOSE 8009

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8009/health')"

CMD ["uvicorn", "ui.main:app", "--host", "0.0.0.0", "--port", "8009"]
