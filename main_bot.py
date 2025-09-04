# main_bot.py — Очищенная и адаптированная версия для Railway

import os
import io
import base64
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Загрузка переменных окружения из .env файла (для локального тестирования)
from dotenv import load_dotenv
load_dotenv()

# Основные библиотеки для работы с API и изображениями
import requests
import aiohttp
from PIL import Image, ImageFilter
import numpy as np
import cv2

# Библиотеки для Telegram-бота (aiogram 3.x)
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, FSInputFile, BufferedInputFile
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import ReplyKeyboardBuilder

# Библиотеки для веб-сервера (вебхук)
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application


# ========= 1. КОНФИГУРАЦИЯ И ПЕРЕМЕННЫЕ =========

# Обязательно укажи эти переменные в настройках Railway
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PIXELCUT_API_KEY = os.getenv("PIXELCUT_API_KEY")
# WEBHOOK_BASE_URL - это публичный URL, который предоставляет Railway
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")

# Необязательные переменные
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
MAIN_BOT_USERNAME = os.getenv("MAIN_BOT_USERNAME", "")

# Проверка наличия обязательных переменных
if not all([BOT_TOKEN, OPENAI_API_KEY, PIXELCUT_API_KEY, WEBHOOK_BASE_URL]):
    raise RuntimeError("Одна или несколько обязательных переменных окружения (BOT_TOKEN, OPENAI_API_KEY, PIXELCUT_API_KEY, WEBHOOK_BASE_URL) не заданы!")

# Настройка логирования для отладки
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# Инициализация объектов aiogram
bot = Bot(BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)


# ========= 2. ТЕКСТЫ И КЛАВИАТУРЫ =========

WELCOME = (
    "👋 Привет! Ты в боте «Предметный фотограф».\n\n"
    "Он поможет:\n"
    "• сделать качественные предметные фото,\n"
    "• заменить фон без потери формы, цвета и надписей,\n"
    "• создать атмосферные сцены (студийно / на человеке / в руках).\n\n"
    "🔐 Чтобы начать, нажми «СТАРТ»."
)

REQUIREMENTS = (
    "📥 Добавь своё фото.\n\n"
    "Требования к исходнику для лучшего результата:\n"
    "• Ровный свет без жёстких теней.\n"
    "• Нейтральный однотонный фон.\n"
    "• Предмет целиком, края не обрезаны.\n"
    "• Максимальное качество (лучше «Документ», чтобы Telegram не сжимал)."
)

PROMPTS_FILE = Path(__file__).parent / "prompts_cheatsheet.md"
PROMPTS_MD = """# 📓 Шпаргалка по промптам для генерации сцен
(сокращено) — опиши фон/свет/настроение, без товара; английский, короткими фразами.
Примеры: studio soft light; dark premium look; glossy marble; cozy interior, warm sunlight; etc.
"""

# ========= Состояния (FSM) =========
class GenStates(StatesGroup):
    waiting_start = State()
    waiting_photo = State()
    waiting_size = State()
    waiting_variants = State()
    waiting_style = State()
    waiting_placement = State()

# ========= Клавиатуры =========
class Placement(str, Enum):
    STUDIO = "Студийно (на фоне)"
    ON_BODY = "На человеке (украшение/одежда)"
    IN_HAND = "В руках (крупный план)"

start_kb = ReplyKeyboardBuilder()
start_kb.button(text="СТАРТ")
start_kb.button(text="📓 Шпаргалка по промтам")
start_kb.adjust(2)
START_KB = start_kb.as_markup(resize_keyboard=True)

place_kb = ReplyKeyboardBuilder()
for p in (Placement.STUDIO.value, Placement.ON_BODY.value, Placement.IN_HAND.value):
    place_kb.button(text=p)
place_kb.adjust(1)
PLACEMENT_KB = place_kb.as_markup(resize_keyboard=True)

PRESETS = [
    "Каталог: чистый студийный фон, мягкая тень",
    "Минимализм: однотон, мягкие тени",
    "Тёмный премиум: low-key, контровый свет",
    "Мрамор/глянец: контролируемые блики",
    "Nature: дерево/лен/зелень, дневной свет",
    "Flat lay: вид сверху, минимум пропсов",
]
style_kb_builder = ReplyKeyboardBuilder()
for p in PRESETS:
    style_kb_builder.button(text=p)
style_kb_builder.button(text="Своя сцена (опишу текстом)")
style_kb_builder.adjust(1)
STYLE_KB = style_kb_builder.as_markup(resize_keyboard=True)

size_kb = ReplyKeyboardBuilder()
for s in ("1:1", "4:5", "3:4", "16:9", "9:16"):
    size_kb.button(text=s)
size_kb.adjust(3, 2)
SIZE_KB = size_kb.as_markup(resize_keyboard=True)

var_kb = ReplyKeyboardBuilder()
for n in ("1", "2", "3", "4", "5"):
    var_kb.button(text=n)
var_kb.adjust(5)
VAR_KB = var_kb.as_markup(resize_keyboard=True)


# ========= 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def ensure_prompts_file():
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.write_text(PROMPTS_MD, encoding="utf-8")

async def download_bytes_from_message(message: Message) -> tuple[bytes, str]:
    if message.document:
        file_id = message.document.file_id
        buf = io.BytesIO()
        await bot.download(message.document, destination=buf)
        return buf.getvalue(), file_id
    elif message.photo:
        file_id = message.photo[-1].file_id
        buf = io.BytesIO()
        await bot.download(message.photo[-1], destination=buf)
        return buf.getvalue(), file_id
    else:
        raise ValueError("В сообщении нет фото/документа")

def ensure_jpg_bytes(image_bytes: bytes) -> bytes:
    """Гарантируем, что изображение в формате RGB JPEG для совместимости с API."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ========= 4. ЛОГИКА ОБРАБОТКИ ИЗОБРАЖЕНИЙ =========

async def remove_bg_pixelcut(image_bytes: bytes) -> bytes:
    """Удаление фона через API Pixelcut."""
    endpoint = "https://api.pixelcut.ai/v1/remove-background"
    headers = {"X-API-Key": PIXELCUT_API_KEY}
    
    data = aiohttp.FormData()
    data.add_field('image',
                   ensure_jpg_bytes(image_bytes),
                   filename='input.jpg',
                   content_type='image/jpeg')

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(endpoint, headers=headers, data=data) as response:
                if response.status == 200:
                    return await response.read()
                elif response.status == 401:
                    raise RuntimeError("Ошибка авторизации (401) в Pixelcut. Проверьте API-ключ.")
                else:
                    detail = await response.text()
                    raise RuntimeError(f"Ошибка API Pixelcut (статус {response.status}): {detail}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Не удалось подключиться к сервису удаления фона: {e}")


def generate_background(prompt: str, size: str) -> Image.Image:
    """Генерация фона через API OpenAI DALL-E 3."""
    endpoint = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "dall-e-3",
        "prompt": f"High-quality product photography background only. {prompt}",
        "size": size,
        "n": 1,
        "quality": "hd",
        "response_format": "b64_json"
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Ошибка генерации фона OpenAI ({response.status_code}): {response.text}")
    
    b64_json = response.json()["data"][0]["b64_json"]
    bg_bytes = base64.b64decode(b64_json)
    return Image.open(io.BytesIO(bg_bytes)).convert("RGBA")


def compose_subject_on_bg(subject_png: bytes, bg_img: Image.Image) -> Image.Image:
    """Наложение объекта на фон с тенью."""
    subj = Image.open(io.BytesIO(subject_png)).convert("RGBA")
    
    # Масштабирование
    canvas_w, canvas_h = bg_img.size
    target_h = int(canvas_h * 0.75)
    scale = target_h / subj.height
    subj = subj.resize((int(subj.width * scale), target_h), Image.LANCZOS)
    
    # Создание тени
    alpha = subj.getchannel('A')
    shadow = Image.new("RGBA", subj.size, (0, 0, 0, 160))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    
    # Композиция
    out = bg_img.copy()
    x = (canvas_w - subj.width) // 2
    y = canvas_h - subj.height - int(canvas_h * 0.05)
    
    out.alpha_composite(shadow, (x + 8, y + 18))
    out.alpha_composite(subj, (x, y))
    return out


def seamless_place(subject_png: bytes, back_img: Image.Image, scale_by_height: float, x_center: int, y_center: int) -> Image.Image:
    """"Бесшовное" встраивание объекта в фон (для рук/тела)."""
    fore = Image.open(io.BytesIO(subject_png)).convert("RGBA")
    back = back_img.convert("RGB")
    
    target_h = int(back.height * scale_by_height)
    ratio = target_h / fore.height
    fore = fore.resize((int(fore.width * ratio), target_h), Image.LANCZOS)
    
    fore_np = cv2.cvtColor(np.array(fore), cv2.COLOR_RGBA2BGRA)
    back_np = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)
    mask = fore_np[:, :, 3]
    
    center = (x_center, y_center)
    
    mixed = cv2.seamlessClone(fore_np[:,:,:3], back_np, mask, center, cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))


# ========= 5. ОБРАБОТЧИКИ СООБЩЕНИЙ (ХЭНДЛЕРЫ) =========

@router.message(CommandStart())
async def on_start(message: Message, state: FSMContext):
    await state.clear()
    ensure_prompts_file()
    await message.answer(WELCOME, reply_markup=START_KB)
    await state.set_state(GenStates.waiting_start)

@router.message(GenStates.waiting_start, F.text == "📓 Шпаргалка по промтам")
async def send_cheatsheet(message: Message):
    try:
        await message.answer_document(FSInputFile(PROMPTS_FILE))
    except Exception:
        await message.answer("⚠️ Файл со шпаргалкой пока недоступен.")

@router.message(GenStates.waiting_start, F.text.casefold() == "старт")
async def pressed_start(message: Message, state: FSMContext):
    await message.answer(REQUIREMENTS)
    await state.set_state(GenStates.waiting_photo)

@router.message(GenStates.waiting_photo, F.photo | F.document)
async def got_photo(message: Message, state: FSMContext):
    try:
        image_bytes, _ = await download_bytes_from_message(message)
        await state.update_data(image=image_bytes)
        await message.answer("Выбери расположение товара:", reply_markup=PLACEMENT_KB)
        await state.set_state(GenStates.waiting_placement)
    except Exception as e:
        logging.error(f"Ошибка обработки фото: {e}")
        await message.answer("Не удалось обработать фото. Попробуй другое.")

@router.message(GenStates.waiting_placement, F.text)
async def choose_placement(message: Message, state: FSMContext):
    await state.update_data(placement=message.text)
    await message.answer("Выбери размер (соотношение сторон):", reply_markup=SIZE_KB)
    await state.set_state(GenStates.waiting_size)

@router.message(GenStates.waiting_size, F.text)
async def choose_size(message: Message, state: FSMContext):
    await state.update_data(size_aspect=message.text)
    await message.answer("Сколько вариантов сделать за один раз?", reply_markup=VAR_KB)
    await state.set_state(GenStates.waiting_variants)

@router.message(GenStates.waiting_variants, F.text)
async def choose_variants(message: Message, state: FSMContext):
    try:
        n = int(message.text or "1")
    except ValueError:
        n = 1
    await state.update_data(n_variants=min(5, max(1, n)))
    await message.answer("Выбери сцену или опиши свою:", reply_markup=STYLE_KB)
    await state.set_state(GenStates.waiting_style)


@router.message(GenStates.waiting_style, F.text)
async def generate_result(message: Message, state: FSMContext):
    style_text = message.text
    await state.update_data(style=style_text)
    await message.answer("Принято! Начинаю магию ✨\nЭто может занять 1-2 минуты...", reply_markup=None)

    try:
        data = await state.get_data()
        image_bytes = data.get("image")
        placement = data.get("placement", Placement.STUDIO.value)
        size_aspect = data.get("size_aspect", "1:1")
        n_variants = data.get("n_variants", 1)
        
        openai_sizes = {"1:1": "1024x1024", "4:5": "1024x1792", "3:4": "1024x1792", "16:9": "1792x1024", "9:16": "1024x1792"}
        openai_size = openai_sizes.get(size_aspect, "1024x1024")

        msg = await message.answer("Шаг 1/3: Удаляю фон с твоего фото...")
        cut_png = await remove_bg_pixelcut(image_bytes)

        for i in range(n_variants):
            await msg.edit_text(f"Шаг 2/3: Генерирую сцену ({i+1}/{n_variants})...")
            
            prompts = {
                Placement.STUDIO.value: f"{style_text}. Studio lighting, photorealistic.",
                Placement.ON_BODY.value: f"{style_text}. Photorealistic human, soft light.",
                Placement.IN_HAND.value: f"{style_text}. Photorealistic hands close-up, soft light."
            }
            bg = generate_background(prompts.get(placement, style_text), size=openai_size)

            await msg.edit_text(f"Шаг 3/3: Совмещаю товар и фон ({i+1}/{n_variants})...")
            
            if placement == Placement.STUDIO.value:
                result = compose_subject_on_bg(cut_png, bg)
            else:
                bw, bh = bg.size
                scale = 0.26 if placement == Placement.ON_BODY.value else 0.40
                center_x = int(bw * 0.5)
                center_y = int(bh * 0.4) if placement == Placement.ON_BODY.value else int(bh * 0.5)
                result = seamless_place(cut_png, bg, scale_by_height=scale, x_center=center_x, y_center=center_y)

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            
            filename = f"result_{i+1}.png"
            await message.answer_document(
                BufferedInputFile(buf.getvalue(), filename),
                caption=f"Вариант {i+1}/{n_variants}"
            )

        await msg.delete()
        await message.answer("✅ Готово!", reply_markup=START_KB)
        await state.set_state(GenStates.waiting_start)

    except Exception as e:
        logging.exception("Ошибка при генерации")
        await message.answer(f"Что-то пошло не так 😥\nОшибка: {e}\n\nПопробуй ещё раз или начни с /start.")
        await state.set_state(GenStates.waiting_photo) # Возвращаем на шаг отправки фото


# ========= 6. ЗАПУСК БОТА (ВЕБХУК) =========

async def on_startup(bot_instance: Bot):
    """Действия при старте приложения."""
    webhook_url = f"{WEBHOOK_BASE_URL}/webhook/{BOT_TOKEN}"
    await bot_instance.set_webhook(webhook_url, drop_pending_updates=True)
    me = await bot.get_me()
    logging.info("Бот @%s запущен через вебхук: %s", me.username, webhook_url)

async def on_shutdown(bot_instance: Bot):
    """Действия при остановке приложения."""
    logging.info("Остановка бота...")
    await bot_instance.delete_webhook()
    await bot.session.close()

def main():
    # Создаем приложение aiohttp
    app = web.Application()
    
    # Регистрируем хэндлеры для старта и остановки
    app.on_startup.append(lambda a: on_startup(bot))
    app.on_shutdown.append(lambda a: on_shutdown(bot))

    # Создаем и регистрируем обработчик вебхуков для aiogram
    webhook_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )
    webhook_handler.register(app, path=f"/webhook/{BOT_TOKEN}")

    # Запускаем приложение
    # Railway сам предоставит нужный PORT в переменной окружения
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

