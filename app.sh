#!/bin/bash
set -e
APP_DIR=/opt/app

# Настройка прав на папку
sudo chown -R ubuntu:ubuntu $APP_DIR

# Создание venv и установка библиотек (если еще нет)
cd $APP_DIR
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# Копирование и запуск сервисов
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable music-api music-bot
sudo systemctl restart music-api music-bot

echo "Services started successfully!"
