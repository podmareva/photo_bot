# main_bot.py — Исправленная и очищенная версия

import os
import io
import base64
import logging
import socket
import asyncio
import ssl
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

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
    Отправляет асинхронный запрос в Pixelcut напрямую по IP,
    чтобы обойти проблемы с DNS на хостинге.
    """
    key = os.getenv("PIXELCUT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PIXELCUT_API_KEY не задан в переменных окружения.")

    _validate_image_bytes(image_bytes)
    jpg_bytes = ensure_jpg_bytes(image_bytes)
    
    hostname = "api.pixelcut.ai"
    # Стабильные IP-адреса Cloudflare, на которых размещен сервис
    ips = ["172.67.175.10", "104.21.58.148"]
    
    data = aiohttp.FormData()
    data.add_field('image',
                   BytesIO(jpg_bytes),
                   filename='input.jpg',
                   content_type='image/jpeg')

    timeout = aiohttp.ClientTimeout(total=120)
    
    # ИЗМЕНЕНО: Создаем SSL-контекст без несовместимого аргумента server_hostname.
    # aiohttp будет использовать заголовок "Host" для проверки сертификата (SNI).
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    last_error = None

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for ip in ips:
            endpoint = f"https://{ip}/v1/remove-background"
            headers = {"X-API-Key": key, "Host": hostname}
            
            try:
                logging.info(f"Pixelcut: Пробую подключиться к {endpoint} (IP: {ip})...")
                async with session.post(endpoint, headers=headers, data=data) as response:
                    if response.status == 200:
                        logging.info("Pixelcut: Фон успешно удален через прямое IP-соединение.")
                        return await response.read()
                    else:
                        detail = await response.text()
                        logging.warning(f"Pixelcut: Ошибка от {ip} (статус {response.status}): {detail}")
                        last_error = RuntimeError(f"Ошибка API (статус {response.status})")
                        if response.status == 401:
                           raise RuntimeError("Ошибка авторизации (401) в Pixelcut. Проверьте правильность вашего API-ключа.")
                        continue # Пробуем следующий IP

            except aiohttp.ClientError as e:
                logging.error(f"Pixelcut: Ошибка соединения с {ip}: {e}")
                last_error = e
                continue # Пробуем следующий IP
    
    # Если все IP-адреса не сработали
    raise RuntimeError(f"Не удалось подключиться к сервису удаления фона: {last_error}")


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

@router.message(GenStates.waiting_start, F.text == "📓 Шпаргалка по промтам")
async def send_cheatsheet(message: Message, state: FSMContext):
    ensure_prompts_file()
    try:
        await message.answer_document(FSInputFile(PROMPTS_FILE))
    except Exception:
        await message.answer("⚠️ Файл со шпаргалкой пока недоступен.")

@router.message(GenStates.waiting_start, F.text.casefold() == "старт")
async def pressed_start(message: Message, state: FSMContext):
    await message.answer(REQUIREMENTS)
    try:
        await message.answer_document(FSInputFile(PROMPTS_FILE), caption="📓 Шпаргалка по промптам")
    except Exception:
        pass
    await message.answer("Пришли фото товара (лучше как Документ).")
    await state.set_state(GenStates.waiting_photo)

@router.message(GenStates.waiting_photo, F.document | F.photo)
async def got_photo(message: Message, state: FSMContext):
    try:
        image_bytes, file_id = await download_bytes_from_message(bot, message)
        await state.update_data(image=image_bytes, image_file_id=file_id)
        await message.answer("Выбери расположение товара:", reply_markup=PLACEMENT_KB)
        await state.set_state(GenStates.waiting_placement)
    except Exception as e:
        logging.error(f"Error processing photo: {e}")
        await message.answer("Не удалось обработать фото. Попробуй другое.")

@router.message(GenStates.waiting_placement, F.text)
async def choose_placement(message: Message, state: FSMContext):
    val = (message.text or "").strip()
    if val not in (Placement.STUDIO.value, Placement.ON_BODY.value, Placement.IN_HAND.value):
        await message.answer("Выбери один из вариантов с помощью клавиатуры.")
        return
    await state.update_data(placement=val)
    await message.answer("Выбери размер (соотношение сторон):", reply_markup=SIZE_KB)
    await state.set_state(GenStates.waiting_size)

@router.message(GenStates.waiting_size, F.text)
async def choose_size(message: Message, state: FSMContext):
    size = (message.text or "").strip()
    if size not in {"1:1", "4:5", "3:4", "16:9", "9:16"}:
        await message.answer("Выбери один из вариантов на клавиатуре.")
        return
    await state.update_data(size_aspect=size)
    await message.answer("Сколько вариантов сделать за один раз?", reply_markup=VAR_KB)
    await state.set_state(GenStates.waiting_variants)

@router.message(GenStates.waiting_variants, F.text)
async def choose_variants(message: Message, state: FSMContext):
    try:
        n = int((message.text or "1").strip())
        n = max(1, min(5, n))
    except ValueError:
        n = 1
    await state.update_data(n_variants=n)
    txt = (
        "Выбери сцену или опиши свою.\n\n"
        "• Studio — clean background, soft gradient, subtle shadow\n"
        "• Lifestyle — warm interior, wood/linen, soft window light\n"
        "• Luxury — glossy stone, controlled highlights, dark backdrop"
    )
    await message.answer(txt, reply_markup=STYLE_KB)
    await state.set_state(GenStates.waiting_style)

@router.message(GenStates.waiting_style, F.text)
async def generate_result(message: Message, state: FSMContext):
    style_text = (message.text or "").strip()
    await state.update_data(style=style_text)
    await message.answer("Принято! Начинаю магию ✨\nЭто может занять 1-2 минуты...", reply_markup=None)

    try:
        data = await state.get_data()
        image_bytes: Optional[bytes] = data.get("image")
        if not image_bytes:
            src_id = data.get("image_file_id")
            if not src_id:
                await message.answer("Не нашел исходное фото. Пожалуйста, начни заново с /start")
                return
            image_bytes = await load_bytes_by_file_id(bot, src_id)
            await state.update_data(image=image_bytes)

        placement = data.get("placement", Placement.STUDIO.value)
        size_aspect = data.get("size_aspect", "1:1")
        n_variants = int(data.get("n_variants", 1))
        openai_size = pick_openai_size(size_aspect)

        # 1) Вырезаем фон
        msg = await message.answer("Шаг 1/3: Удаляю фон с твоего фото...")
        cut_png = await remove_bg_pixelcut(image_bytes)

        for i in range(n_variants):
            await msg.edit_text(f"Шаг 2/3: Генерирую сцену ({i+1}/{n_variants})...")
            # 2) Генерируем фон
            if placement == Placement.STUDIO.value:
                prompt = f"{style_text}. Background only, no product. photorealistic, studio lighting, realistic textures, no text."
            elif placement == Placement.ON_BODY.value:
                prompt = f"{style_text}. photorealistic human portrait, neutral background, visible neck and collarbone, soft diffused light, shallow depth of field, natural skin tones, allow central empty area for necklace, no text."
            else:  # IN_HAND
                prompt = f"{style_text}. photorealistic hands close-up, neutral background, soft window light, macro-friendly composition, allow central empty area for product, no text."

            bg = generate_background(prompt, size=openai_size)
            bg = center_crop_to_aspect(bg, size_aspect)

            await msg.edit_text(f"Шаг 3/3: Совмещаю товар и фон ({i+1}/{n_variants})...")
            # 3) Компонуем
            if placement == Placement.STUDIO.value:
                result = compose_subject_on_bg(cut_png, bg, scale_by_height=0.74, x_shift=0.0, y_shift=-0.06)
            elif placement == Placement.ON_BODY.value:
                bw, bh = bg.size
                result = seamless_place(cut_png, bg, scale_by_height=0.26, x=int(bw*0.5), y=int(bh*0.38))
            else: # IN_HAND
                bw, bh = bg.size
                result = seamless_place(cut_png, bg, scale_by_height=0.40, x=int(bw*0.5), y=int(bh*0.5))

            # 4) Отправляем результат
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            data_bytes = buf.getvalue()
            filename = f"result_{i+1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            await message.answer_document(BufferedInputFile(data_bytes, filename), caption=f"Вариант {i+1}/{n_variants}")

        await msg.delete()
        await state.clear()
        await message.answer("✅ Готово! Можешь прислать следующее фото или нажми /start для смены настроек.", reply_markup=START_KB)
        await state.set_state(GenStates.waiting_start)

    except Exception as e:
        logging.exception("Generation error")
        await message.answer(f"Что-то пошло не так 😥\nОшибка: {e}\n\nПопробуй ещё раз или начни с /start.")
        # Стейт не чистим, чтобы можно было повторить/исправить

# === Webhook server ===
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
BASE_URL = (os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
assert BASE_URL, "WEBHOOK_BASE_URL или RENDER_EXTERNAL_URL должны быть заданы"
WEBHOOK_URL = BASE_URL + WEBHOOK_PATH

async def on_startup_app(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
    await _log_bot_info()

async def on_shutdown_app(app: web.Application):
    await bot.delete_webhook()
    await bot.session.close()

async def albato_generate_handler(request):
    try:
        data = await request.json()
        file_id = data.get("file_id")
        style = data.get("style", "studio soft light")
        aspect = data.get("aspect", "1:1")
        placement = data.get("placement", "Студийно (на фоне)")

        if not file_id:
            return web.json_response({"error": "file_id is required"}, status=400)

        image_bytes = await load_bytes_by_file_id(bot, file_id)
        cut_png = await remove_bg_pixelcut(image_bytes)

        prompt = style
        if placement == "Студийно (на фоне)":
            prompt += ". Background only, no product. photorealistic, studio lighting, realistic textures, no text."
        elif placement == "На человеке (украшение/одежда)":
            prompt += ". photorealistic human portrait, neutral background, soft diffused light, no text."
        else:
            prompt += ". photorealistic hands close-up, neutral background, macro-friendly composition, no text."

        bg = generate_background(prompt, size=pick_openai_size(aspect))
        bg = center_crop_to_aspect(bg, aspect)

        if placement == "Студийно (на фоне)":
            result = compose_subject_on_bg(cut_png, bg, scale_by_height=0.74, x_shift=0.0, y_shift=-0.06)
        elif placement == "На человеке (украшение/одежда)":
            bw, bh = bg.size
            result = seamless_place(cut_png, bg, scale_by_height=0.26, x=int(bw*0.5), y=int(bh*0.38))
        else:
            bw, bh = bg.size
            result = seamless_place(cut_png, bg, scale_by_height=0.40, x=int(bw*0.5), y=int(bh*0.5))

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        return web.json_response({"image_base64": encoded_image})

    except Exception as e:
        logging.exception("Albato Webhook Error")
        return web.json_response({"error": str(e)}, status=500)

app = web.Application()
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
setup_application(app, dp, on_startup=on_startup_app, on_shutdown=on_shutdown_app)

app.router.add_post("/albato-generate", albato_generate_handler)

print("✅ Запуск main_bot.py")
print("BOT_TOKEN:", BOT_TOKEN[:10], "...")
print("BASE_URL:", BASE_URL)
print("WEBHOOK_URL:", WEBHOOK_URL)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
