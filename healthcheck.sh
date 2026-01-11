#!/bin/bash
set -euo pipefail

cd /opt/app

if ! curl -f http://localhost:8000/health &> /dev/null; then
    echo "[ALERT] API is down! Restarting..."
    sudo docker-compose restart api
    exit 1
fi

if ! sudo docker ps | grep -q music-bot; then
    echo "[ALERT] Bot is down! Restarting..."
    sudo docker-compose restart bot
    exit 1
fi

echo "[OK] All services healthy"
exit 0
