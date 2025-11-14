FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEBDRIVER_MANAGER_CACHE=/tmp/wdm \
    CHROME_BIN=/usr/bin/chromium \
    CHROMIUM_FLAGS="--no-sandbox --disable-dev-shm-usage"

WORKDIR /app

COPY Craiglist-Scrapper/requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    chromium-sandbox \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnss3 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxkbcommon0 \
    libxrandr2 \
    wget \
    ca-certificates \
    findutils \
&& rm -rf /var/lib/apt/lists/* \
&& (find /usr -name "chromedriver" -type f 2>/dev/null | head -1 | xargs -I {} ln -sf {} /usr/bin/chromedriver || true) \
&& (find /usr -name "chromedriver" -type f 2>/dev/null | head -1 | xargs -I {} chmod +x {} || true) \
&& chmod +x /usr/bin/chromedriver 2>/dev/null || true \
&& chmod 4755 /usr/lib/chromium/chromium-sandbox 2>/dev/null || true \
&& chmod 4755 /usr/lib/chromium-browser/chromium-sandbox 2>/dev/null || true \
&& pip install --upgrade pip \
&& pip install -r requirements.txt

COPY Craiglist-Scrapper /app/Craiglist-Scrapper

ENV PYTHONPATH=/app/Craiglist-Scrapper

EXPOSE 8080

CMD ["python", "Craiglist-Scrapper/messaging.py", "server"]

