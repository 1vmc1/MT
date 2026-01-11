#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}[app.sh] Starting Docker deployment...${NC}"

cd /opt/app || { echo -e "${RED}[ERROR] /opt/app not found${NC}"; exit 1; }

if [[ ! -f .env ]]; then
    echo -e "${RED}[ERROR] .env file not found!${NC}"
    exit 1
fi

# Остановка старых контейнеров
echo -e "${YELLOW}[app.sh] Stopping old containers...${NC}"
docker compose down --remove-orphans 2>/dev/null || true

# Загрузка свежих образов из Docker Hub
echo -e "${YELLOW}[app.sh] Pulling latest images from Docker Hub...${NC}"
docker compose pull

# Запуск контейнеров
echo -e "${YELLOW}[app.sh] Starting containers...${NC}"
docker compose up -d

# Ожидание запуска
echo -e "${YELLOW}[app.sh] Waiting for services...${NC}"
sleep 15

# Статус контейнеров
echo -e "${GREEN}[app.sh] Container status:${NC}"
docker compose ps

# Проверка здоровья API
echo -e "${YELLOW}[app.sh] Checking API health...${NC}"
for i in {1..10}; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "${GREEN}✓ API is healthy${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}✗ API health check failed after 10 attempts${NC}"
        exit 1
    else
        echo "Attempt $i/10 failed, retrying in 3s..."
        sleep 3
    fi
done

# Логи
echo -e "${GREEN}[app.sh] Recent logs:${NC}"
echo -e "${YELLOW}=== API logs ===${NC}"
docker logs --tail 30 music-api 2>/dev/null || echo "music-api container not found"

echo -e "${YELLOW}=== BOT logs ===${NC}"
docker logs --tail 30 music-bot 2>/dev/null || echo "music-bot container not found"

echo -e "${GREEN}[app.sh] Deployment complete!${NC}"
