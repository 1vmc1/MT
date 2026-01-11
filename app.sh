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

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[app.sh] Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}[app.sh] Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo -e "${YELLOW}[app.sh] Stopping old containers...${NC}"
sudo docker-compose down --remove-orphans || true

echo -e "${YELLOW}[app.sh] Building and starting containers...${NC}"
sudo docker-compose build --no-cache
sudo docker-compose up -d

echo -e "${YELLOW}[app.sh] Waiting for services...${NC}"
sleep 15

echo -e "${GREEN}[app.sh] Container status:${NC}"
sudo docker-compose ps

if curl -f http://localhost:8000/health &> /dev/null; then
    echo -e "${GREEN}✓ API is healthy${NC}"
else
    echo -e "${RED}✗ API health check failed${NC}"
fi

echo -e "${GREEN}[app.sh] Deployment complete!${NC}"
sudo docker logs --tail 20 music-api
sudo docker logs --tail 20 music-bot
