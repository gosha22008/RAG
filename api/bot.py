"""Telegram-бот для RAG-системы.
Запуск: python api/bot.py
"""
import asyncio
import logging
import os

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("RAG_API_URL", "http://localhost:10000")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Привет! Я бот, который отвечает на вопросы по документации.\n\n"
        "Просто задай мне вопрос — я найду ответ в документах и пришлю его."
    )


@dp.message(Command("help"))
async def help_cmd(message: Message):
    await message.answer(
        "Как пользоваться:\n"
        "Просто напиши вопрос — я отвечу по документации.\n\n"
        "Пример: «Какое действие считается моментом Активации ПВЗ?»"
    )


@dp.message(F.text)
async def handle_question(message: Message):
    query = message.text.strip()
    if not query:
        return

    # Показываем «печатает»
    await bot.send_chat_action(message.chat.id, "typing")
    thinking_msg = await message.answer("🔍 Ищу ответ...")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_URL}/ask",
                json={"text": query},
            )
            response.raise_for_status()
            data = response.json()

        answer = data.get("answer", "Не удалось получить ответ.")
        pages = data.get("pages", [])

        # Собираем уникальные страницы
        all_pages = sorted({p for sublist in pages for p in sublist})
        pages_str = ", ".join(all_pages) if all_pages else "—"

        text = (
            f"💡 *Ответ:*\n{answer}\n\n"
            # f"📄 *Страницы документа:* {pages_str}"
        )

        await thinking_msg.edit_text(text, parse_mode="Markdown")

    except httpx.HTTPError as e:
        logger.error(f"API error: {e}")
        await thinking_msg.edit_text("⚠️ Сервер RAG недоступен. Попробуй позже.")
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        await thinking_msg.edit_text(f"⚠️ Ошибка: {e}")


async def main():
    logger.info(f"Bot starting, API: {API_URL}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())