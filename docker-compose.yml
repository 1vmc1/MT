version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: music-classifier:latest
    container_name: music-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - PYTHONUNBUFFERED=1
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  bot:
    build:
      context: .
      dockerfile: Dockerfile
    image: music-classifier:latest
    container_name: music-bot
    restart: unless-stopped
    command: ["python", "bot.py"]
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - API_URL=http://api:8000
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    depends_on:
      api:
        condition: service_healthy
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
