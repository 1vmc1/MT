#!/bin/bash
set -euo pipefail

APP_DIR=/opt/app
API_SERVICE=music-api
BOT_SERVICE=music-bot

echo "[app.sh] cd ${APP_DIR}"
cd "${APP_DIR}"

echo "[app.sh] create venv"
python3 -m venv .venv

echo "[app.sh] install requirements"
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "[app.sh] install systemd units"
sudo cp -f systemd/*.service /etc/systemd/system/

echo "[app.sh] reload systemd"
sudo systemctl daemon-reload

echo "[app.sh] enable + restart services"
sudo systemctl enable "${API_SERVICE}" "${BOT_SERVICE}"
sudo systemctl restart "${API_SERVICE}" "${BOT_SERVICE}"

echo "[app.sh] check services are active"
sleep 2

if ! systemctl is-active --quiet "${API_SERVICE}"; then
  echo "[app.sh] ERROR: ${API_SERVICE} is not active"
  sudo systemctl status "${API_SERVICE}" --no-pager || true
  sudo journalctl -u "${API_SERVICE}" -n 200 --no-pager || true
  exit 1
fi

if ! systemctl is-active --quiet "${BOT_SERVICE}"; then
  echo "[app.sh] ERROR: ${BOT_SERVICE} is not active"
  sudo systemctl status "${BOT_SERVICE}" --no-pager || true
  sudo journalctl -u "${BOT_SERVICE}" -n 200 --no-pager || true
  exit 1
fi

echo "[app.sh] OK: both services are active"
sudo systemctl status "${API_SERVICE}" --no-pager || true
sudo systemctl status "${BOT_SERVICE}" --no-pager || true
