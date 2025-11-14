FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEBDRIVER_MANAGER_CACHE=/tmp/wdm

WORKDIR /app

COPY Craiglist-Scrapper/requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
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
&& rm -rf /var/lib/apt/lists/* \
&& pip install --upgrade pip \
&& pip install -r requirements.txt

COPY Craiglist-Scrapper /app/Craiglist-Scrapper

ENV PYTHONPATH=/app/Craiglist-Scrapper

EXPOSE 8080

CMD ["python", "Craiglist-Scrapper/messaging.py", "server"]

