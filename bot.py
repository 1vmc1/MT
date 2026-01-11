
# bot.py
from __future__ import annotations

import os
import asyncio
import logging
from pathlib import Path
import json
import aiohttp

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("music_bot")

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
API_URL = os.getenv("API_URL", "http://localhost:8000/predict_segmented")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# Метки
try:
    with open("labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
        label2idx = labels.get("label2idx", {})
        idx2label = {int(k): v for k, v in labels.get("idx2label", {}).items()}
    logger.info("labels.json загружен")
except Exception as e:
    label2idx = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4,
                 "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}
    idx2label = {v: k for k, v in label2idx.items()}
    logger.warning(f"labels.json не найден/повреждён, использую дефолт. Детали: {e}")

def guess_ext_by_mime(mime: str | None) -> str:
    mt = (mime or "").lower()
    if mt in ("audio/mpeg", "audio/mp3"):
        return "mp3"
    if mt in ("audio/wav", "audio/x-wav"):
        return "wav"
    if mt in ("audio/ogg", "audio/opus", "audio/x-opus+ogg"):
        return "ogg"
    if mt == "audio/webm":
        return "webm"
    if mt.startswith("audio/"):
        return mt.split("/", 1)[1]
    return "bin"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [KeyboardButton("Загрузить файл")],
        [KeyboardButton("О системе"), KeyboardButton("Сообщить об ошибке")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    # Сбросим режим ожидания на старте, чтобы точно быть в "чистом" состоянии
    context.user_data.pop("awaiting_error_genre", None)
    await update.message.reply_text(
        "🎵 Добро пожаловать в Music Genre Classifier!\n\n"
        "Пришлите аудио (MP3/WAV/OGG, до 20 МБ) — определю жанр.",
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Отправьте аудиофайл (MP3/WAV/OGG), максимум {MAX_FILE_SIZE//(1024*1024)} МБ.\n"
        "Я верну жанр и уверенность."
    )

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # сбрасываем режим ожидания, если пользователь запросил другую команду
    context.user_data.pop("awaiting_error_genre", None)
    await update.message.reply_text(
        "Поддерживаемые жанры:\n- " + "\n- ".join(label2idx.keys())
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # получение аудио тоже должно прерывать режим ожидания ошибки
    context.user_data.pop("awaiting_error_genre", None)

    temp_path = None
    try:
        msg = update.message
        file_entity = msg.audio or msg.voice or msg.document
        if not file_entity:
            await msg.reply_text("Пришлите аудио/голос/документ с аудио.")
            return

        if getattr(file_entity, "file_size", None) and file_entity.file_size > MAX_FILE_SIZE:
            await msg.reply_text("❌ Файл слишком большой (лимит 20 МБ).")
            return

        tg_file = await context.bot.get_file(file_entity.file_id)

        # Расширение
        ext = None
        if tg_file.file_path and "." in tg_file.file_path:
            ext = Path(tg_file.file_path).suffix.lstrip(".").lower()
        if not ext:
            ext = guess_ext_by_mime(getattr(file_entity, "mime_type", None)) or "bin"

        temp_path = f"temp_{getattr(file_entity, 'file_unique_id', 'file')}.{ext}"
        await tg_file.download_to_drive(temp_path)

        # POST в API
        form_data = aiohttp.FormData()
        content_type = getattr(file_entity, "mime_type", None) or "application/octet-stream"
        with open(temp_path, "rb") as f:
            form_data.add_field("file", f, filename=Path(temp_path).name, content_type=content_type)
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, data=form_data) as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                       # Для /predict_segmented
                        genre = payload.get("overall_genre")  or "unknown"
                        conf = payload.get("overall_confidence") or "unknown"
                        conf_txt = f"{float(conf):.2%}" if isinstance(conf, (int, float)) else None
                        text = f"🎧 Жанр: {genre}"
                        if conf_txt:
                             text += f"\n📈 Уверенность: {conf_txt}"
                        await msg.reply_text(text)
                    else:
                        err = await resp.text()
                        logger.error(f"API {resp.status}: {err}")
                        await msg.reply_text("❌ Ошибка API.")
    except Exception as e:
        logger.exception("Ошибка в обработчике", exc_info=e)
        await update.message.reply_text("❌ Ошибка при обработке файла.")
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = (update.message.text or "").strip()
    t_lower = t.lower()

    # Если мы в режиме ожидания жанра от пользователя (после "Сообщить об ошибке")
    if context.user_data.get("awaiting_error_genre"):
        # принимаем жанр, никуда не сохраняем
        genre_from_user = t  # сохраняем в локальную переменную, но не в файл/БД
        context.user_data.pop("awaiting_error_genre", None)
        await update.message.reply_text(f"Спасибо! Жанр \"{genre_from_user}\" принят. Мы разберёмся и свяжемся при необходимости.")
        return

    # Обычная обработка кнопок/текста
    if t_lower == "загрузить файл":
        # при выборе другого пункта сбрасываем режим ожидания
        context.user_data.pop("awaiting_error_genre", None)
        await update.message.reply_text("Пришлите аудиофайл (MP3/WAV/OGG).")
    elif t_lower == "о системе":
        await info_command(update, context)
    elif t_lower == "сообщить об ошибке":
        # переводим пользователя в режим ожидания жанра (никуда не сохраняем)
        context.user_data["awaiting_error_genre"] = True
        await update.message.reply_text("Пожалуйста, укажите жанр, в котором вы обнаружили ошибку. Просто отправьте название жанра.")
    else:
        await update.message.reply_text("Не понял. Пришлите аудиофайл или используйте кнопки меню.")

async def modelinfo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(API_URL.replace("/predict_segmented", "/")) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    model_type = data.get("model_type", "неизвестно")
                    weights = data.get("weights", "неизвестно")
                    device = data.get("device", "неизвестно")
                    text = (
                        f"Информация о модели:\n"
                        f"Тип модели: {model_type}\n"
                        f"Веса: {weights}\n"
                        f"Устройство: {device}"
                    )
                else:
                    text = f"Ошибка при запросе информации о модели: HTTP {resp.status}"
    except Exception as e:
        text = f"Ошибка при запросе информации о модели: {e}"
    await update.message.reply_text(text)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Исключение в боте", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("❌ Произошла ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("info", info_command))
    app.add_handler(CommandHandler("modelinfo", modelinfo_command))

    audio_filter = (
        filters.AUDIO
        | filters.VOICE
        | filters.Document.MimeType("audio/mpeg")
        | filters.Document.MimeType("audio/x-wav")
        | filters.Document.MimeType("audio/wav")
        | filters.Document.MimeType("audio/ogg")
        | filters.Document.MimeType("audio/webm")
        | filters.Document.FileExtension("mp3")
        | filters.Document.FileExtension("wav")
        | filters.Document.FileExtension("ogg")
        | filters.Document.FileExtension("webm")
    )
    app.add_handler(MessageHandler(audio_filter, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)

    logger.info(f"Старт бота. API_URL={API_URL}")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()