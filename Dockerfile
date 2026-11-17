FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PONDLITE_ENV=production

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# PondLite currently hardcodes ROOT/pondlite.db. Keep that path stable while
# storing the actual database in the separately mounted state volume.
RUN groupadd --system pondlite \
    && useradd --system --gid pondlite --home-dir /app pondlite \
    && mkdir -p /app/data /app/runs /app/raw /app/state \
    && rm -f /app/pondlite.db \
    && ln -s /app/state/pondlite.db /app/pondlite.db \
    && chown -R pondlite:pondlite /app

USER pondlite

EXPOSE 8420 8421
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8420/health', timeout=3).read()" || exit 1

CMD ["bash", "scripts/start-production.sh"]
