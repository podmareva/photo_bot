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
from PIL import Image, ImageFilter
import numpy as np
import cv2

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    FSInputFile,
    BufferedInputFile,
)
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart
from aiogram import Router
from aiogram.utils.keyboard import ReplyKeyboardBuilder

# webhook + aiohttp server
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from io import BytesIO
import requests

import os, requests

PIXELCUT_API_KEY = os.getenv("PIXELCUT_API_KEY", "").strip()

def build_pixelcut_headers() -> dict:
    """
    Если ключ похож на JWT (три части через точки) — шлём как Bearer.
    Иначе используем X-API-KEY.
    """
    if PIXELCUT_API_KEY.count(".") == 2:  # Header.Payload.Signature
        return {"Authorization": f"Bearer {PIXELCUT_API_KEY}"}
    else:
        return {"X-API-KEY": PIXELCUT_API_KEY}


# ========= ENV / CONFIG =========
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PIXELCUT_API_KEY = os.getenv("PIXELCUT_API_KEY")
PIXELCUT_ENDPOINT = os.getenv(
    "PIXELCUT_ENDPOINT", "https://api.developer.pixelcut.ai/v1/remove-background"
)
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
MAIN_BOT_USERNAME = os.getenv("MAIN_BOT_USERNAME", "")

assert BOT_TOKEN, "BOT_TOKEN is required"

# === OPTIONAL PG (сохраняем только метаданные/file_id) ===
import psycopg2

def db_exec(q: str, params: tuple = ()):  # очень простой helper
    if not DATABASE_URL:
        return None
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(q, params)
            if cur.description:
                return cur.fetchall()
    finally:
        conn.close()

def gallery_save(
    user_id: int,
    src_file_id: str,
    cut_file_id: str,
    placement: str,
    size_aspect: str,
    style_text: str,
    n_variants: int,
    result_file_ids: List[str],
):
    if not DATABASE_URL:
        return
    db_exec(
        """INSERT INTO items(user_id, src_file_id, cut_file_id, placement, size_aspect, style_text, n_variants, result_file_ids)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            user_id,
            src_file_id,
            cut_file_id,
            placement,
            size_aspect,
            style_text,
            n_variants,
            result_file_ids,
        ),
    )

def gallery_last(user_id: int):
    if not DATABASE_URL:
        return None
    rows = db_exec(
        """SELECT id, src_file_id, cut_file_id, placement, size_aspect, style_text
           FROM items WHERE user_id=%s ORDER BY created_at DESC LIMIT 1""",
        (user_id,),
    )
    return rows[0] if rows else None

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

# ====== Содержимое файла prompts_cheatsheet.md (если его нет — создадим) ======
PROMPTS_MD = """# 📓 Шпаргалка по промптам для генерации сцен

Общее правило: описывай только фон/окружение/свет/настроение. Не упоминай сам товар.
Промпты — на английском (точнее для модели). Делай короткие фразы через запятую.
Если важно сохранить вид товара: add — `do not change product color, shape or details`.

---

## 1) Чистый каталог / студия
- white seamless background, soft studio lighting, natural shadows — чистый белый фон, мягкий свет
- light gradient background, minimalism, soft shadows — светлый градиент, минимализм
- clean pastel background, centered composition, no props — пастельный фон, без реквизита

## 2) Минимализм
- matte single-color background, pastel tones, soft shadows — матовый однотон, мягкие тени
- light concrete wall, soft diffused light — светлый бетон, рассеянный свет
- beige background, airy atmosphere, no props — бежевый фон, «воздух», без реквизита

## 3) Тёмный премиум / драматичный
- deep black background, dramatic rim light, high contrast — чёрный фон, контровый свет
- dark gradient background, soft highlights, premium look — тёмный градиент, мягкие блики
- black velvet texture, macro shot, controlled reflections — чёрный бархат, контролируемые отражения

## 4) Глянец / камень / мрамор
- glossy marble surface, dark background, soft studio light — глянцевый мрамор, тёмный фон
- black granite surface, focused light — чёрный гранит, направленный свет
- mirror reflection, product on glass, moody lighting — стекло и отражение, атмосферный свет

## 5) Косметика / уход
- frosted glass surface, gradient background, soft glow — матовое стекло, градиент, свечение
- acrylic stand, warm diffused light — акриловая подставка, тёплый рассеянный свет
- mirror tiles, clean pastel background, soft highlights — зеркальная плитка, пастель

## 6) Натуральные материалы
- light wood table, soft daylight, eucalyptus leaves — светлое дерево, дневной свет, зелень
- linen fabric folds, warm side light — лён, складки ткани, тёплый боковой
- stone and wood surface, morning sunlight — камень+дерево, утреннее солнце

## 7) Интерьер
- cozy living room, wooden furniture, warm sunlight from window — уютная гостиная, тёплый свет из окна
- modern kitchen, clean surfaces, soft daylight — современная кухня, чистые плоскости
- spa-style bathroom, stone, greenery, steam glow — спа-ванная, камень, паровое свечение

## 8) Украшение на человеке (автоген)
- photorealistic human portrait, neutral background, visible neck and collarbone, soft diffused light, shallow depth of field, natural skin tones — портрет, видна шея/ключицы, мягкий свет
- beauty close-up, neutral background, film grain, warm tones — бьюти-крупный план, нейтральный фон
- editorial style portrait, soft backlight glow, minimal makeup — фэшн-портрет, мягкая подсветка

## 9) В руках (автоген)
- photorealistic hands close-up, neutral background, soft window light, macro-friendly composition — крупный план рук, мягкий свет из окна
- female hands, natural skin texture, shallow depth of field — женские руки, естественная кожа, малая ГРИП
- hands holding space, warm interior bokeh, cozy mood — руки с «пустым местом», тёплое боке

## 10) Сезоны
**Лето**
- sunlight, leaf shadows, warm tones — солнечный свет, тени листвы
- beach sand, soft waves, bright sky — пляж, волны, яркое небо

**Осень**
- golden hour light, autumn leaves, cozy atmosphere — золотой час, листья, уют
- wooden table, pumpkins, warm side light — стол, тыквы, тёплый боковой

**Зима**
- snow-covered branches, cold blue light — снег, холодный голубой свет
- cozy interior, fairy lights, Christmas mood — уют, гирлянды, новый год

**Весна**
- fresh greenery, blooming branches, soft sunlight — свежая зелень, цветы, мягкое солнце
- pastel background, gentle glow — пастель, мягкое свечение

## 11) Праздники / огни
- warm bokeh lights, dark background, cozy mood — тёплое боке на тёмном
- bright garlands, festive atmosphere — яркие гирлянды, праздник
- fireworks background, high contrast — фейерверки, контраст

## 12) Flat Lay (вид сверху)
- top view, matte surface, soft light, minimal props — вид сверху, матовая поверхность
- pastel background, neat composition — пастель, аккуратная раскладка
- wooden table, props around edges, soft shadows — дерево, реквизит по краям

## 13) Техно / индустриальный
- smooth concrete, cold directional light, graphic shadows — гладкий бетон, холодный свет
- metallic surface, reflections, blue highlights — металл, отражения, синие акценты
- neon accents, black background — неон, чёрный фон

## 14) Усилители качества (добавляй в конец)
- photorealistic, ultra detailed, 8k
- studio softbox lighting, realistic textures
- centered composition, generous negative space
- natural soft shadows
- no props, no text
"""

# ========= states =========
class GenStates(StatesGroup):
    waiting_start = State()
    waiting_photo = State()
    waiting_service = State()
    waiting_placement = State()
    waiting_size = State()
    waiting_variants = State()
    waiting_style = State()

# ========= choices/keyboards =========
class CutService(str, Enum):
    REMBG = "Эконом (RemBG — бесплатно)"
    PIXELCUT = "Премиум (Pixelcut — лучше качество)"

class Placement(str, Enum):
    STUDIO = "Студийно (на фоне)"
    ON_BODY = "На человеке (украшение/одежда)"
    IN_HAND = "В руках (крупный план)"

# Кнопка СТАРТ + шпаргалка
start_kb = ReplyKeyboardBuilder()
start_kb.button(text="СТАРТ")
start_kb.button(text="📓 Шпаргалка по промтам")
start_kb.adjust(2)
START_KB = start_kb.as_markup(resize_keyboard=True)

# выбор сервиса вырезки
cut_kb = ReplyKeyboardBuilder()
cut_kb.button(text=CutService.REMBG.value)
cut_kb.button(text=CutService.PIXELCUT.value)
cut_kb.adjust(1)
CUT_KB = cut_kb.as_markup(resize_keyboard=True)

# расположение
place_kb = ReplyKeyboardBuilder()
for p in (Placement.STUDIO.value, Placement.ON_BODY.value, Placement.IN_HAND.value):
    place_kb.button(text=p)
place_kb.adjust(1)
PLACEMENT_KB = place_kb.as_markup(resize_keyboard=True)

# пресеты стиля
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

# размеры и количество
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
OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1/images/generations"

def ensure_prompts_file():
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.write_text(PROMPTS_MD, encoding="utf-8")

async def check_user_access(user_id: int) -> bool:
    if ADMIN_ID and user_id == ADMIN_ID:
        return True
    return True  # пускаем всех

async def download_bytes_from_message(bot: Bot, message: Message) -> tuple[bytes, str]:
    """Скачать байты файла из сообщения + вернуть file_id для галереи."""
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
	
def remove_bg_rembg_bytes(image_bytes: bytes) -> bytes:
    try:
        from rembg import remove, new_session
    except Exception as e:
        raise RuntimeError(f"rembg недоступен: {e}")
    try:
        session = new_session("u2netp")     # меньше RAM и быстрее
        return remove(image_bytes, session=session)
    except Exception as e:
        raise RuntimeError(f"Ошибка rembg: {e}")

import asyncio

async def remove_bg_rembg_bytes_async(image_bytes: bytes) -> bytes:
    # переносим тяжёлую работу в поток
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, remove_bg_rembg_bytes, image_bytes)

async def generate_with_rembg_or_timeout(image_bytes: bytes) -> bytes:
    try:
        return await asyncio.wait_for(remove_bg_rembg_bytes_async(image_bytes), timeout=40)
    except asyncio.TimeoutError:
        raise RuntimeError("rembg: истек таймаут 40с. Попробуйте Премиум или другое фото.")

import os, requests

def build_pixelcut_headers() -> dict:
    key = os.getenv("PIXELCUT_API_KEY", "").strip()
    if key.count(".") == 2:               # JWT: Header.Payload.Signature
        return {"Authorization": f"Bearer {key}"}
    return {"X-API-KEY": key}

def remove_bg_pixelcut(image_bytes: bytes) -> bytes:
    endpoint = os.getenv("PIXELCUT_ENDPOINT")
    if not endpoint:
        raise RuntimeError("PIXELCUT_ENDPOINT не задан")

    headers = build_pixelcut_headers()
    jpg = ensure_jpg_bytes(image_bytes)

    def call(field_name: str) -> bytes:
        files = {field_name: ("input.jpg", BytesIO(jpg), "image/jpeg")}
        r = requests.post(endpoint, headers=headers, files=files, timeout=120)
        if r.status_code == 200:
            return r.content
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"Ошибка Pixelcut: {r.status_code}: {detail}")

    # Автоподбор имени поля файла — частая причина 400
    for field in ("image", "image_file", "file", "file_upload"):
        try:
            return call(field)
        except RuntimeError as e:
            msg = str(e)
            if ("Unsupported format" in msg
                or "invalid_parameter" in msg
                or "unsupported" in msg.lower()):
                continue
            raise
    raise RuntimeError("Pixelcut: ни одно из имён полей не подошло")

def pick_openai_size(aspect: str) -> str:
    if aspect == "1:1":
        return "1024x1024"
    if aspect in ("4:5", "3:4", "9:16"):
        return "1024x1792"
    if aspect == "16:9":
        return "1792x1024"
    return "1024x1024"

def center_crop_to_aspect(img: Image.Image, aspect: str) -> Image.Image:
    w, h = img.size
    targets = {"1:1": 1/1, "4:5": 4/5, "3:4": 3/4, "16:9": 16/9, "9:16": 9/16}
    if aspect not in targets:
        return img
    r = targets[aspect]
    cur = w / h
    if abs(cur - r) < 1e-3:
        return img
    if cur > r:
        new_w = int(h * r)
        x1 = (w - new_w) // 2
        return img.crop((x1, 0, x1 + new_w, h))
    else:
        new_h = int(w / r)
        y1 = (h - new_h) // 2
        return img.crop((0, y1, w, y1 + new_h))

def generate_background(prompt: str, size: str = "1024x1024") -> Image.Image:
    assert OPENAI_API_KEY, "OPENAI_API_KEY is required"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-image-1",
        "prompt": (
            "High-quality product photography background only (no product). "
            "Cinematic lighting, realistic textures. " + prompt
        ),
        "size": size,
        "n": 1,
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
    subj = subj.resize((max(1, int(subj.width * scale)), max(1, int(subj.height * scale))), Image.LANCZOS)
    alpha = subj.split()[-1]
    shadow = Image.new("RGBA", subj.size, (0, 0, 0, 160))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    out = bg_img.copy()
    x = int((canvas_w - subj.width) * 0.5 + x_shift * canvas_w)
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
    fore_rgb = cv2.cvtColor(np.array(fore.convert("RGB")), cv2.COLOR_RGB2BGR)
    back_bgr = cv2.cvtColor(np.array(back), cv2.COLOR_RGB2BGR)
    mask = np.array(fore.split()[-1])
    fh, fw = fore_rgb.shape[:2]
    x = max(0, min(back_bgr.shape[1] - fw, x))
    y = max(0, min(back_bgr.shape[0] - fh, y))
    canvas = np.zeros_like(back_bgr)
    canvas[y:y+fh, x:x+fw] = fore_rgb
    mask_full = np.zeros(back_bgr.shape[:2], dtype=np.uint8)
    mask_full[y:y+fh, x:x+fw] = mask
    center = (x + fw // 2, y + fh // 2)
    mixed = cv2.seamlessClone(canvas, back_bgr, mask_full, center, cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))

def ensure_jpg_bytes(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

# ========= handlers =========
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
    image_bytes, file_id = await download_bytes_from_message(bot, message)
    await state.update_data(image=image_bytes, image_file_id=file_id)
    await message.answer("Чем вырезать фон?", reply_markup=CUT_KB)
    await state.set_state(GenStates.waiting_service)

@router.message(GenStates.waiting_service, F.text)
async def choose_service(message: Message, state: FSMContext):
    choice = (message.text or "").strip()
    if choice not in (CutService.REMBG.value, CutService.PIXELCUT.value):
        await message.answer("Выбери вариант на клавиатуре.")
        return
    await state.update_data(cut_service=choice)
    await message.answer("Выбери расположение товара:", reply_markup=PLACEMENT_KB)
    await state.set_state(GenStates.waiting_placement)

@router.message(GenStates.waiting_placement, F.text)
async def choose_placement(message: Message, state: FSMContext):
    val = (message.text or "").strip()
    if val not in (Placement.STUDIO.value, Placement.ON_BODY.value, Placement.IN_HAND.value):
        await message.answer("Выбери один из вариантов.")
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
    except ValueError:
        n = 1
    n = max(1, min(5, n))
    await state.update_data(n_variants=n)
    txt = (
        "Выбери сцену или опиши свою.\n\n"
        "• Studio — clean background, soft gradient, subtle shadow\n"
        "• Lifestyle — warm interior, wood/linen, soft window light\n"
        "• Luxury — glossy stone, controlled highlights, dark backdrop\n\n"
        "Совет: исходники как Документ — Telegram не сжимает."
    )
    await message.answer(txt, reply_markup=STYLE_KB)
    await state.set_state(GenStates.waiting_style)

@router.message(GenStates.waiting_style, F.text)
async def generate_result(message: Message, state: FSMContext):
    style_text = (message.text or "").strip()
    await state.update_data(style=style_text)
    await message.answer("Генерирую…")

    try:
        data = await state.get_data()
        image_bytes: Optional[bytes] = data.get("image")
        if image_bytes is None:
            src_id = data.get("image_file_id")
            if not src_id:
                await message.answer("Нет исходника — начни со /start")
                return
            image_bytes = await load_bytes_by_file_id(bot, src_id)
            await state.update_data(image=image_bytes)

        cut_service = data.get("cut_service", CutService.REMBG.value)
        placement = data.get("placement", Placement.STUDIO.value)
        size_aspect = data.get("size_aspect", "1:1")
        n_variants = int(data.get("n_variants", 1))

        openai_size = pick_openai_size(size_aspect)

        # 1) вырезаем фон (один раз)
        if cut_service == CutService.PIXELCUT.value:
            cut_png = remove_bg_pixelcut(image_bytes)
        else:
            cut_png = remove_bg_rembg_bytes(image_bytes)

        result_file_ids: List[str] = []

        for i in range(n_variants):
            # 2) генерируем фон
            if placement == Placement.STUDIO.value:
                prompt = (
                    f"{style_text}. Background only, no product. "
                    "photorealistic, studio lighting, realistic textures, no text."
                )
            elif placement == Placement.ON_BODY.value:
                prompt = (
                    f"{style_text}. photorealistic human portrait, neutral background, "
                    "visible neck and collarbone, soft diffused light, shallow depth of field, "
                    "natural skin tones, allow central empty area for necklace, no text."
                )
            else:  # IN_HAND
                prompt = (
                    f"{style_text}. photorealistic hands close-up, neutral background, soft window light, "
                    "macro-friendly composition, allow central empty area for product, no text."
                )

            bg = generate_background(prompt, size=openai_size)
            bg = center_crop_to_aspect(bg, size_aspect)

            # 3) компоновка
            if placement == Placement.STUDIO.value:
                result = compose_subject_on_bg(cut_png, bg, scale_by_height=0.74, x_shift=0.0, y_shift=-0.06)
            elif placement == Placement.ON_BODY.value:
                bw, bh = bg.size
                x = bw // 2 - 1
                y = int(bh * 0.38)
                result = seamless_place(cut_png, bg, scale_by_height=0.26, x=x, y=y)
            else:
                bw, bh = bg.size
                x = bw // 2 - 1
                y = int(bh * 0.5)
                result = seamless_place(cut_png, bg, scale_by_height=0.40, x=x, y=y)

            # 4) отправка
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            data_bytes = buf.getvalue()
            filename = f"product_{i+1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            await message.answer_document(BufferedInputFile(data_bytes, filename), caption=f"Вариант {i+1}/{n_variants}")

        # 5) сохранить запись в галерее (метаданные и file_id — если нужно, перехватывай file_id через send result)
        try:
            src_id = data.get("image_file_id", "")
            gallery_save(
                message.from_user.id,
                src_file_id=src_id,
                cut_file_id="",
                placement=placement,
                size_aspect=size_aspect,
                style_text=style_text,
                n_variants=n_variants,
                result_file_ids=result_file_ids,
            )
        except Exception as e:
            logging.warning("Gallery save failed: %s", e)

    except Exception as e:
        logging.exception("Generation error")
        await message.answer(f"Ошибка: {e}")
    finally:
        await state.clear()
        await message.answer("Готово. Пришли ещё фото или /start.")

@router.message(F.text == "/repeat")
async def repeat_last(message: Message, state: FSMContext):
    row = gallery_last(message.from_user.id)
    if not row:
        await message.answer("В галерее пока пусто. Сначала сгенерируй фото.")
        return
    _id, src_file_id, cut_file_id, placement, size_aspect, style_text = row
    await state.update_data(
        image_file_id=src_file_id,
        placement=placement,
        size_aspect=size_aspect,
        n_variants=1,
    )
    await message.answer("Повторим. Выбери стиль/пресет или напиши свой промпт.", reply_markup=STYLE_KB)
    await state.set_state(GenStates.waiting_style)

# === Webhook server (единый) ===
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
BASE_URL = (os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL", "")).rstrip("/")
assert BASE_URL, "WEBHOOK_BASE_URL или RENDER_EXTERNAL_URL должны быть заданы"
WEBHOOK_URL = BASE_URL + WEBHOOK_PATH

async def on_startup_app(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)

async def on_shutdown_app(app: web.Application):
    await bot.delete_webhook()
    await bot.session.close()

app = web.Application()
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
setup_application(app, dp, on_startup=on_startup_app, on_shutdown=on_shutdown_app)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
