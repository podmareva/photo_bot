import os
import io
import base64
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
from PIL import Image, ImageFilter
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils.executor import start_webhook

# == OPTIONAL local free remover ==
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# == OpenCV для мягкого вписывания ==
import numpy as np
import cv2

# ====== CONFIG ======
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PIXELCUT_API_KEY = os.getenv("PIXELCUT_API_KEY")
PIXELCUT_ENDPOINT = os.getenv(
    "PIXELCUT_ENDPOINT", "https://api.developer.pixelcut.ai/v1/remove-background"
)

# опционально
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
MAIN_BOT_USERNAME = os.getenv("MAIN_BOT_USERNAME", "")

# ====== DB helpers (галерея хранит только метаданные и file_id) ======
import psycopg2

def db_exec(q, params=()):
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
    result_file_ids: list,
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

assert BOT_TOKEN, "BOT_TOKEN is required"
assert OPENAI_API_KEY, "OPENAI_API_KEY is required"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

import os as _os, logging as _logging
_logging.info("Bot starting… PID=%s, instance=%s",
             _os.getpid(), _os.getenv("RENDER_INSTANCE_ID"))

async def _log_bot_info():
    me = await bot.get_me()
    logging.info("Bot: @%s (id=%s)", me.username, me.id)

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

# ====== STATES ======
class GenStates(StatesGroup):
    waiting_start = State()
    waiting_photo = State()
    waiting_service = State()
    waiting_placement = State()  # студия / на человеке / в руках (автоген)
    waiting_size = State()       # выбор соотношения сторон
    waiting_variants = State()   # сколько вариантов
    waiting_style = State()

# ====== CHOICES & KEYBOARDS ======
class CutService(str, Enum):
    REMBG = "Эконом (RemBG — бесплатно)"
    PIXELCUT = "Премиум (Pixelcut — лучше качество)"

class Placement(str, Enum):
    STUDIO = "Студийно (на фоне)"
    ON_BODY = "На человеке (украшение/одежда)"
    IN_HAND = "В руках (крупный план)"

start_kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
start_kb.add(types.KeyboardButton("СТАРТ"))
start_kb.add(types.KeyboardButton("📓 Шпаргалка по промтам"))

CUT_KB = types.ReplyKeyboardMarkup(resize_keyboard=True)
CUT_KB.add(CutService.REMBG.value)
CUT_KB.add(CutService.PIXELCUT.value)

PLACEMENT_KB = types.ReplyKeyboardMarkup(resize_keyboard=True)
PLACEMENT_KB.add(Placement.STUDIO.value)
PLACEMENT_KB.add(Placement.ON_BODY.value)
PLACEMENT_KB.add(Placement.IN_HAND.value)

PRESETS = [
    "Каталог: чистый студийный фон, мягкий градиент, аккуратная тень",
    "Минимализм: однотонный матовый фон, мягкие тени",
    "Светлый монохром: high-key, ровный дневной свет",
    "Тёмный монохром: low-key, глубокие тени, контровый свет",
    "Luxury: глянцевый камень/мрамор, контролируемые блики",
    "Nature-mood: дерево, лен, зелень, рассеянный свет",
    "Flat lay: вид сверху, минимальные пропсы",
    "Косметика: матовый акрил, стекло, мягкие отражения",
    "Украшения: бархат, макро-свет, контролируемые блики",
    "Еда/выпечка: деревянный стол, тёплый утренний свет",
    "Техника: бетон/алюминий, холодный свет, геометрия",
    "Праздничный: нейтральный фон, тёплое боке огней",
    "Лето/аутдор: тёплый солнечный свет, тени листвы",
    "Камень/мрамор: полированный мрамор, мягкие блики",
    "Бетон: гладкий серый бетон, графичные тени",
    "Лён/текстиль: мягкие складки, дневной свет",
]
STYLE_KB = types.ReplyKeyboardMarkup(resize_keyboard=True)
for p in PRESETS:
    STYLE_KB.add(types.KeyboardButton(p))
STYLE_KB.add(types.KeyboardButton("Своя сцена (опишу текстом)"))

OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1/images/generations"

SIZE_KB = types.ReplyKeyboardMarkup(resize_keyboard=True)
for s in ["1:1", "4:5", "3:4", "16:9", "9:16"]:
    SIZE_KB.add(types.KeyboardButton(s))

VAR_KB = types.ReplyKeyboardMarkup(resize_keyboard=True)
for n in ["1", "2", "3", "4", "5"]:
    VAR_KB.add(types.KeyboardButton(n))

# ===== ACCESS =====
def check_user_access(user_id: int) -> bool:
    if ADMIN_ID and user_id == ADMIN_ID:
        return True
    return True  # для тестов пускаем всех

# ====== HELPERS ======
async def load_bytes_by_file_id(bot: Bot, file_id: str) -> bytes:
    """Скачать байты исходника по Telegram file_id (нужно для /repeat)."""
    f = await bot.get_file(file_id)
    fb = await bot.download_file(f.file_path)
    return fb.read()

def ensure_prompts_file():
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.write_text(PROMPTS_MD, encoding="utf-8")

def remove_bg_rembg(image_bytes: bytes) -> bytes:
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg не установлен. Установи: pip install rembg")
    return rembg_remove(
        image_bytes,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5,
    )

def remove_bg_pixelcut(image_bytes: bytes) -> bytes:
    if not PIXELCUT_API_KEY or not PIXELCUT_ENDPOINT:
        raise RuntimeError("Не задан PIXELCUT_API_KEY или PIXELCUT_ENDPOINT")
    headers = {"X-API-KEY": PIXELCUT_API_KEY}
    files = {
        "image": ("image.png", image_bytes, "image/png"),
        "image_file": ("image.png", image_bytes, "image/png"),
    }
    resp = requests.post(PIXELCUT_ENDPOINT, headers=headers, files=files, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Pixelcut error {resp.status_code}: {resp.text}")
    ctype = resp.headers.get("Content-Type", "")
    if "image/" in ctype or resp.content.startswith(b"\x89PNG") or resp.content.startswith(b"\xff\xd8"):
        return resp.content
    try:
        data = resp.json()
        url = (
            data.get("result_url")
            or data.get("url")
            or (data.get("data", {}).get("url") if isinstance(data.get("data"), dict) else None)
        )
        if not url:
            raise ValueError("Не найдено поле result_url/url в ответе Pixelcut")
        img = requests.get(url, timeout=90)
        img.raise_for_status()
        return img.content
    except Exception:
        raise RuntimeError(f"Неизвестный формат ответа Pixelcut: {resp.text[:400]}")

def pick_openai_size(aspect: str) -> str:
    # gpt-image-1: 1024x1024, 1024x1792 (портрет ~9:16), 1792x1024 (ландшафт ~16:9)
    if aspect == "1:1":
        return "1024x1024"
    if aspect in ("4:5", "3:4", "9:16"):
        return "1024x1792"   # портрет
    if aspect == "16:9":
        return "1792x1024"   # горизонталь
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
    """Простая компоновка + мягкая тень (без cv2)."""
    subj = Image.open(io.BytesIO(subject_png)).convert("RGBA")
    canvas_w, canvas_h = bg_img.size
    target_h = int(canvas_h * scale_by_height)
    scale = target_h / max(1, subj.height)
    subj = subj.resize(
        (max(1, int(subj.width * scale)), max(1, int(subj.height * scale))),
        Image.LANCZOS,
    )

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
    """
    Мягко «вживляет» вырезанный предмет в фон (cv2.seamlessClone).
    x, y — позиция левого-верхнего угла прямоугольника вставки (центр будет рассчитан).
    """
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

# ====== BOT HANDLERS ======
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message, state: FSMContext):
    if not check_user_access(message.from_user.id):
        await message.answer(f"⛔ Нет доступа. Получите доступ через @{MAIN_BOT_USERNAME}.")
        return
    await state.finish()
    ensure_prompts_file()
    await message.answer(WELCOME, reply_markup=start_kb)
    await GenStates.waiting_start.set()

@dp.message_handler(lambda m: (m.text or "").strip().upper() == "СТАРТ", state=GenStates.waiting_start)
async def on_press_start(message: types.Message, state: FSMContext):
    await message.answer(REQUIREMENTS, reply_markup=types.ReplyKeyboardRemove())
    try:
        with open(PROMPTS_FILE, "rb") as f:
            await message.answer_document(
                types.InputFile(f, filename=PROMPTS_FILE.name),
                caption="📓 Шпаргалка по промптам",
            )
    except FileNotFoundError:
        await message.answer("⚠️ Шпаргалка пока недоступна.")
    await message.answer("Пришли фото товара (лучше как Документ).")
    await GenStates.waiting_photo.set()

@dp.message_handler(state=GenStates.waiting_start, regexp="^📓 Шпаргалка по промтам$")
async def send_cheatsheet(message: types.Message, state: FSMContext):
    ensure_prompts_file()
    try:
        with open(PROMPTS_FILE, "rb") as f:
            await message.answer_document(
                types.InputFile(f, filename=PROMPTS_FILE.name),
                caption="📓 Шпаргалка по промптам",
            )
    except FileNotFoundError:
        await message.answer("⚠️ Файл со шпаргалкой не найден.")

@dp.message_handler(state=GenStates.waiting_photo, content_types=["photo", "document"])
async def got_photo(message: types.Message, state: FSMContext):
    # Получаем file_id для галереи и байты для вырезки
    if message.document:
        src_file_id = message.document.file_id
        f = await bot.get_file(message.document.file_id)
    else:
        src_file_id = message.photo[-1].file_id
        f = await bot.get_file(message.photo[-1].file_id)
    fb = await bot.download_file(f.file_path)
    await state.update_data(image=fb.read(), image_file_id=src_file_id)

    await message.answer("Чем вырезать фон?", reply_markup=CUT_KB)
    await GenStates.waiting_service.set()

@dp.message_handler(state=GenStates.waiting_service, content_types=["text"])
async def choose_service(message: types.Message, state: FSMContext):
    choice = (message.text or "").strip()
    if choice not in (CutService.REMBG.value, CutService.PIXELCUT.value):
        await message.answer("Выбери вариант на клавиатуре.")
        return
    await state.update_data(cut_service=choice)

    await message.answer("Выбери расположение товара:", reply_markup=PLACEMENT_KB)
    await GenStates.waiting_placement.set()

@dp.message_handler(state=GenStates.waiting_placement, content_types=["text"])
async def choose_placement(message: types.Message, state: FSMContext):
    val = (message.text or "").strip()
    if val not in (Placement.STUDIO.value, Placement.ON_BODY.value, Placement.IN_HAND.value):
        await message.answer("Выбери один из вариантов.")
        return
    await state.update_data(placement=val)

    await message.answer("Выбери размер (соотношение сторон):", reply_markup=SIZE_KB)
    await GenStates.waiting_size.set()

@dp.message_handler(state=GenStates.waiting_size, content_types=["text"])
async def choose_size(message: types.Message, state: FSMContext):
    size = (message.text or "").strip()
    if size not in {"1:1", "4:5", "3:4", "16:9", "9:16"}:
        await message.answer("Выбери один из вариантов на клавиатуре.")
        return
    await state.update_data(size_aspect=size)
    await message.answer("Сколько вариантов сделать за один раз?", reply_markup=VAR_KB)
    await GenStates.waiting_variants.set()

@dp.message_handler(state=GenStates.waiting_variants, content_types=["text"])
async def choose_variants(message: types.Message, state: FSMContext):
    try:
        n = int((message.text or "1").strip())
    except ValueError:
        n = 1
    n = max(1, min(5, n))
    await state.update_data(n_variants=n)
    txt = (
        "Выбери сцену или опиши свою.\n\n"
        "Подсказки по промптам:\n"
        "• Каталог — clean studio background, soft gradient, subtle shadow.\n"
        "• Lifestyle — warm interior corner, wood, linen, soft window light.\n"
        "• Luxury — glossy stone, controlled highlights, dark backdrop.\n\n"
        "Совет: отправляй исходники как *Документ* — Telegram не сжимает фото."
    )
    await message.answer(txt, reply_markup=STYLE_KB, parse_mode="Markdown")
    await GenStates.waiting_style.set()

@dp.message_handler(state=GenStates.waiting_style, content_types=["text"])
async def got_style(message: types.Message, state: FSMContext):
    style_text = (message.text or "").strip()
    await state.update_data(style=style_text)
    await message.answer("Генерирую…", reply_markup=types.ReplyKeyboardRemove())

    try:
        data = await state.get_data()

        # 0) подтянуть байты исходника по file_id, если нужно (для /repeat)
        image_bytes = data.get("image")
        if image_bytes is None:
            src_id = data.get("image_file_id")
            if not src_id:
                raise RuntimeError("Нет исходника: загрузите фото или используйте /start.")
            image_bytes = await load_bytes_by_file_id(bot, src_id)
            await state.update_data(image=image_bytes)

        cut_service = data["cut_service"]
        placement = data.get("placement", Placement.STUDIO.value)
        size_aspect = data.get("size_aspect", "1:1")
        n_variants = int(data.get("n_variants", 1))

        openai_size = pick_openai_size(size_aspect)

        # 1) вырезаем один раз
        if cut_service == CutService.PIXELCUT.value:
            cut_png = remove_bg_pixelcut(image_bytes)
        else:
            cut_png = remove_bg_rembg(image_bytes)

        result_file_ids = []

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
                result = compose_subject_on_bg(
                    cut_png, bg, scale_by_height=0.74, x_shift=0.0, y_shift=-0.06
                )
            elif placement == Placement.ON_BODY.value:
                bw, bh = bg.size
                x = bw // 2 - 1
                y = int(bh * 0.38)  # зона шеи
                result = seamless_place(cut_png, bg, scale_by_height=0.26, x=x, y=y)
            else:  # IN_HAND
                bw, bh = bg.size
                x = bw // 2 - 1
                y = int(bh * 0.5)   # центр
                result = seamless_place(cut_png, bg, scale_by_height=0.40, x=x, y=y)

            # 4) отправка
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            filename = f"product_{i+1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            sent = await message.answer_document(
                types.InputFile(buf, filename=filename),
                caption=f"Вариант {i+1}/{n_variants}",
            )
            if sent and sent.document:
                result_file_ids.append(sent.document.file_id)

        # 5) сохранить запись в галерее (метаданные и file_id)
        try:
            src_id = data.get("image_file_id", "")
            gallery_save(
                message.from_user.id,
                src_file_id=src_id,
                cut_file_id="",  # вырезку в ТГ не отправляем — пусто
                placement=placement,
                size_aspect=size_aspect,
                style_text=style_text,
                n_variants=n_variants,
                result_file_ids=result_file_ids,
            )
        except Exception as e:
            logging.warning(f"Gallery save failed: {e}")

    except Exception as e:
        logging.exception("Generation error")
        await message.answer(f"Ошибка: {e}")

    await state.finish()
    await message.answer("Пришли ещё фото или /start.")

@dp.message_handler(commands=["repeat"])
async def repeat_last(message: types.Message, state: FSMContext):
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
    await message.answer(
        "Повторная генерация с последними настройками. Выбери стиль/пресет или напиши свой промпт.",
        reply_markup=STYLE_KB,
    )
    await GenStates.waiting_style.set()

# --- webhook config for Render (aiogram v2) ---
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")
assert WEBHOOK_BASE_URL, "WEBHOOK_BASE_URL is required"
WEBHOOK_URL = f"{WEBHOOK_BASE_URL}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.environ.get("PORT", 10000))

async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    await _log_bot_info()
    logging.info("Webhook set: %s", WEBHOOK_URL)

async def on_shutdown(dp):
    logging.info("Shutting down…")
    await bot.delete_webhook()

if __name__ == "__main__":
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
