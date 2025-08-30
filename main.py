import logging
import os
import json
import requests # Импортируем синхронную библиотеку для HTTP-запросов
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# --- НАСТРОЙКА И ИНИЦИАЛИЗАЦИЯ ---

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальный словарь для хранения историй чатов {chat_id: [messages]}
chat_histories = {}

def load_system_prompt() -> dict | None:
    """Загружает системный промпт из файла system_prompt.json."""
    try:
        with open("system_prompt.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Ошибка загрузки системного промпта: {e}")
        return None

SYSTEM_PROMPT = load_system_prompt() # Загружаем промпт при старте

# --- ВЗАИМОДЕЙСТВИЕ С API LLM ---

def get_llm_response(api_key: str, history: list) -> str | None:
    """
    Отправляет синхронный запрос к API OpenRouter и возвращает ответ модели.
    ВНИМАНИЕ: Этот запрос блокирует основной поток, пока не будет получен ответ.
    Для высоконагруженных ботов лучше использовать асинхронные библиотеки (например, httpx).
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.5-flash",
                "messages": history,
            }
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка API OpenRouter: {e}")
        return None

# --- ОБРАБОТЧИКИ TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение."""
    welcome_message = (
        "Привет! Я бот, подключенный к Large Language Model через OpenRouter. "
        "Отправь мне любое сообщение, чтобы начать общение."
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения пользователя."""
    chat_id = update.effective_chat.id
    user_message = update.message.text

    # 1. Получаем или инициализируем историю для данного чата
    history = chat_histories.get(chat_id)
    if history is None:
        history = []
        if SYSTEM_PROMPT:
            history.append(SYSTEM_PROMPT)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Критическая ошибка: Системный промпт не загружен."
            )
            return
    
    # 2. Добавляем сообщение пользователя в историю
    history.append({"role": "user", "content": user_message})

    # 3. Получаем API ключ
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.error("OPENROUTER_API_KEY не найден.")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Ошибка: Ключ API для OpenRouter не настроен."
        )
        return

    # 4. Получаем ответ от LLM
    # Запускаем синхронную функцию в асинхронном контексте
    import asyncio
    llm_response = await asyncio.get_event_loop().run_in_executor(
        None, get_llm_response, openrouter_api_key, history
    )

    # 5. Обрабатываем ответ
    if llm_response:
        history.append({"role": "assistant", "content": llm_response})
        # Ограничиваем контекст до 20 последних сообщений
        if len(history) > 20:
            # Сохраняем системный промпт + 19 последних сообщений
            history = [history[0]] + history[-19:]
        
        chat_histories[chat_id] = history
        await context.bot.send_message(chat_id=chat_id, text=llm_response)
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Извините, произошла ошибка при обращении к модели."
        )

# --- ЗАПУСК БОТА ---

def main():
    """Основная функция для запуска бота."""
    if not SYSTEM_PROMPT:
        logger.critical("Не удалось загрузить системный промпт. Запуск бота отменен.")
        return

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        logger.critical("TELEGRAM_BOT_TOKEN не найден. Запуск бота отменен.")
        return

    application = ApplicationBuilder().token(telegram_token).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Бот запускается...")
    application.run_polling()

if __name__ == '__main__':
    main()
