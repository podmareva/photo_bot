# main_bot.py — Исправленная и очищенная версия

import os
import io
import base64
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

import requests
import aiohttp # Добавлена новая библиотека для запросов
from PIL import Image, ImageFilter
import numpy as np
import cv2

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, FSInputFile, BufferedInputFile
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from io import BytesIO

# ========= ENV / CONFIG =========
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PIXELCUT_API_KEY = os.getenv("PIXELCUT_API_KEY", "").strip() # ВАЖНО: Убедись, что этот ключ правильный!
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
MAIN_BOT_USERNAME = os.getenv("MAIN_BOT_USERNAME", "")

assert BOT_TOKEN, "BOT_TOKEN is required"

# ========= logging =========
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

bot = Bot(BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)

async def _log_bot_info():
    me = await bot.get_me()
    logging.info("Bot: @%s (%s)", me.username, me.id)

# ===== TEXTS =====
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

# ========= states =========
class GenStates(StatesGroup):
    waiting_start = State()
    waiting_photo = State()
    waiting_size = State()
    waiting_variants = State()
    waiting_style = State()
    waiting_placement = State()

# ========= choices/keyboards =========
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

# ========= helpers =========
def ensure_prompts_file():
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.write_text(PROMPTS_MD, encoding="utf-8")

async def check_user_access(user_id: int) -> bool:
    if ADMIN_ID and user_id == ADMIN_ID:
        return True
    return True  # пускаем всех

async def download_bytes_from_message(bot: Bot, message: Message) -> tuple[bytes, str]:
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

async def load_bytes_by_file_id(bot: Bot, file_id: str) -> bytes:
    buf = io.BytesIO()
    await bot.download(file_id, destination=buf)
    return buf.getvalue()

# ====== Pixelcut API Calls ======

def _validate_image_bytes(image_bytes: bytes) -> None:
    """Проверка, что отправляем реальную картинку, иначе Pixelcut вернёт 400."""
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 1024:
        raise RuntimeError("Исходный файл пустой или слишком маленький (<1 КБ)")
    try:
        Image.open(BytesIO(image_bytes)).verify()
    except Exception as e:
        raise RuntimeError(f"Файл не распознан как изображение: {e}")

def ensure_jpg_bytes(image_bytes: bytes) -> bytes:
    """Гарантируем корректный RGB-JPEG без альфы для внешнего API."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()

async def remove_bg_pixelcut(image_bytes: bytes) -> bytes:
    """
    Отправляет асинхронный запрос в Pixelcut с использованием aiohttp
    для лучшей работы в облачных средах.
    """
    key = os.getenv("PIXELCUT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PIXELCUT_API_KEY не задан в переменных окружения.")

    _validate_image_bytes(image_bytes)
    jpg_bytes = ensure_jpg_bytes(image_bytes)

    endpoint = "https://api.pixelcut.ai/v1/remove-background"
    headers = {"X-API-Key": key}
    
    data = aiohttp.FormData()
    data.add_field('image',
                   BytesIO(jpg_bytes),
                   filename='input.jpg',
                   content_type='image/jpeg')

    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            logging.info(f"Pixelcut: Отправляю aiohttp запрос на {endpoint}...")
            async with session.post(endpoint, headers=headers, data=data) as response:
                if response.status == 200:
                    logging.info("Pixelcut: Фон успешно удален через aiohttp.")
                    return await response.read()
                else:
                    try:
                        detail = await response.json()
                    except Exception:
                        detail = await response.text()
                    
                    error_message = f"Ошибка от API Pixelcut (статус {response.status}): {detail}"
                    logging.error(error_message)
                    
                    if response.status == 401:
                        raise RuntimeError("Ошибка авторизации (401) в Pixelcut. Проверьте правильность вашего API-ключа.")
                    
                    raise RuntimeError(error_message)

        except aiohttp.ClientError as e:
            logging.error(f"Сетевая ошибка aiohttp при обращении к Pixelcut: {e}")
            raise RuntimeError(f"Не удалось подключиться к сервису удаления фона: {e}")


# ====== Image Generation ======

OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1/images/generations"

def pick_openai_size(aspect: str) -> str:
    if aspect == "1:1": return "1024x1024"
    if aspect in ("4:5", "3:4", "9:16"): return "1024x1792"
    if aspect == "16:9": return "1792x1024"
    return "1024x1024"

def center_crop_to_aspect(img: Image.Image, aspect: str) -> Image.Image:
    w, h = img.size
    targets = {"1:1": 1.0, "4:5": 4/5, "3:4": 3/4, "16:9": 16/9, "9:16": 9/16}
    if aspect not in targets: return img
    
    target_ratio = targets[aspect]
    current_ratio = w / h
    
    if abs(current_ratio - target_ratio) < 1e-3: return img

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return img.crop((x1, 0, x1 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        return img.crop((0, y1, w, y1 + new_h))

def generate_background(prompt: str, size: str = "1024x1024") -> Image.Image:
    assert OPENAI_API_KEY, "OPENAI_API_KEY is required"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "dall-e-3", # Рекомендуется использовать актуальную модель
        "prompt": (
            "High-quality product photography background only (no product). "
            "Cinematic lighting, realistic textures. " + prompt
        ),
        "size": size,
        "n": 1,
        "quality": "hd",
        "response_format": "b64_json"
    }
    r = requests.post(OPENAI_IMAGES_ENDPOINT, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI image gen error {r.status_code}: {r.text}")
    
    b64 = r.json()["data"][0]["b64_json"]
    bg_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(bg_bytes)).convert("RGBA")

def compose_subject_on_bg(
    subject_png: bytes,
    bg_img: Image.Image,
    *,
    scale_by_height: float = 0.75,
    x_shift: float = 0.0,
    y_shift: float = -0.05,
) -> Image.Image:
    subj = Image.open(io.BytesIO(subject_png)).convert("RGBA")
    canvas_w, canvas_h = bg_img.size
    target_h = int(canvas_h * scale_by_height)
    scale = target_h / max(1, subj.height)
    subj = subj.resize((max(1, int(subj.width * scale)), target_h), Image.LANCZOS)
    
    # Создание тени
    alpha = subj.split()[-1]
    shadow = Image.new("RGBA", subj.size, (0, 0, 0, 160))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    
    out = bg_img.copy()
    x = int((canvas_w - subj.width) / 2 + x_shift * canvas_w)
    y = int(canvas_h - subj.height + y_shift * canvas_h)
    
    out.alpha_composite(shadow, (x + 8, y + 18))
    out.alpha_composite(subj, (x, y))
    return out

def seamless_place(
    subject_png: bytes,
    back_img: Image.Image,
    *,
    scale_by_height: float,
    x: int,
    y: int,
) -> Image.Image:
    fore = Image.open(io.BytesIO(subject_png)).convert("RGBA")
    back = back_img.convert("RGB")
    
    bw, bh = back.size
    target_h = int(bh * scale_by_height)
    ratio = target_h / max(1, fore.height)
    fore = fore.resize((max(1, int(fore.width * ratio)), target_h), Image.LANCZOS)
    
    fore_rgb_np = cv2.cvtColor(np.array(fore.convert("RGB")), cv2.COLOR_RGB2BGR)
    back_bgr_np = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)
    mask = np.array(fore.split()[-1])
    
    fh, fw = fore_rgb_np.shape[:2]
    # Корректное определение центра для вставки
    center_x = x + fw // 2
    center_y = y + fh // 2

    # Убедимся, что объект не выходит за границы фона
    x = max(0, min(bw - fw, x))
    y = max(0, min(bh - fh, y))

    # Создаем холст для объекта и полную маску
    obj_canvas = np.zeros_like(back_bgr_np)
    obj_canvas[y:y+fh, x:x+fw] = fore_rgb_np
    
    mask_full = np.zeros(back_bgr_np.shape[:2], dtype=np.uint8)
    mask_full[y:y+fh, x:x+fw] = mask
    
    mixed = cv2.seamlessClone(fore_rgb_np, back_bgr_np, mask, (center_x, center_y), cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))

# ========= Handlers =========
@router.message(CommandStart())
async def on_start(message: Message, state: FSMContext):
    if not await check_user_access(message.from_user.id):
        await message.answer(f"⛔ Нет доступа. Получите доступ через @{MAIN_BOT_USERNAME}.")
        return
    await state.clear()
    ensure_prompts_file()
    await message.answer(WELCOME, reply_markup=START_KB)
    await state.set_state(GenStates.waiting_start)

@router.message(GenStates.waiting_start, F.text == "📓 Шпаргалка по п
