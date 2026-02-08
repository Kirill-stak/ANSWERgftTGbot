import os
import re
import telebot
from telebot import types
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ===========================================
# НАСТРОЙКИ
# ===========================================
BOT_TOKEN = ""  # ← ЗАМЕНИТЕ НА СВОЙ ТОКЕН

if not BOT_TOKEN:
    raise ValueError("Укажите токен бота в переменной BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

# Загрузка модели Qwen2.5 (только один раз при старте)
print("Загружаю модель Qwen2.5-1.5B-Instruct... (~3 ГБ, 2-3 минуты)")
device = 0 if torch.cuda.is_available() else -1

# Явная загрузка для корректной работы с чат-шаблоном
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto" if device == 0 else "cpu",
    torch_dtype=torch.float16 if device == 0 else torch.float32,
    low_cpu_mem_usage=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32
)
print(f"✅ Модель Qwen2.5 загружена. Устройство: {'GPU' if device == 0 else 'CPU'}")

# Состояние пользователей: {user_id: {'recipient': '...', 'budget': '...'}}
user_state = {}

# ===========================================
# КЛАВИАТУРЫ
# ===========================================
def get_main_keyboard():
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("🎁 Помоги выбрать подарок", callback_data="start_flow"))
    return markup

def get_recipient_keyboard():
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("Другу 👨", callback_data="recipient:другу"),
        types.InlineKeyboardButton("Подруге 👩", callback_data="recipient:подруге"),
        types.InlineKeyboardButton("Программисту 💻", callback_data="recipient:программисту")
    )
    return markup

def get_budget_keyboard():
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("До 100 ₽", callback_data="budget:100"),
        types.InlineKeyboardButton("2 500–3 000 ₽", callback_data="budget:2500-3000"),
        types.InlineKeyboardButton("5 000–15 000 ₽", callback_data="budget:5000-15000"),
        types.InlineKeyboardButton("30 000–150 000 ₽", callback_data="budget:30000-150000")
    )
    return markup

# ===========================================
# ГЕНЕРАЦИЯ ПОДАРКОВ (чат-формат для Qwen2.5)
# ===========================================
budget_map = {
    "100": "до 100 рублей",
    "2500-3000": "2500–3000 рублей",
    "5000-15000": "5000–15000 рублей",
    "30000-150000": "30000–150000 рублей"
}

def generate_gift_suggestion(recipient: str, budget_code: str) -> str:
    """Генерация подарка через чат-промпт для Qwen2.5"""
    budget_text = budget_map.get(budget_code, budget_code)
    
    # Системный промпт для управления поведением модели
    messages = [
        {"role": "system", "content": (
            "Ты — эксперт по подаркам. Отвечай ТОЛЬКО на русском языке. "
            "Предложи ОДИН конкретный, практичный и оригинальный подарок для указанного человека и бюджета. "
            "Ответ должен быть кратким — одно предложение без лишних комментариев, вопросов или оговорок. "
            "Не пиши 'Я предлагаю', 'Можно подарить' — сразу назови подарок."
        )},
        {"role": "user", "content": f"Подарок для {recipient} с бюджетом {budget_text}."}
    ]
    
    try:
        outputs = pipe(
            messages,
            max_new_tokens=60,
            temperature=0.65,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Извлекаем ответ ассистента (последнее сообщение в чате)
        generated_messages = outputs[0]["generated_text"]
        assistant_reply = ""
        for msg in reversed(generated_messages):
            if msg["role"] == "assistant":
                assistant_reply = msg["content"].strip()
                break
        
        if not assistant_reply:
            return get_fallback_suggestion(recipient, budget_code)
        
        # Очистка ответа: первое осмысленное предложение
        # Удаляем типичные "отказы" модели
        refusal_patterns = [
            r"извини", r"не могу", r"не умею", r"я — языковая модель", 
            r"я не могу", r"я не имею возможности", r"я не должен",
            r"я не рекомендую", r"я не советую", r"я не предлагаю"
        ]
        if any(re.search(pattern, assistant_reply.lower()) for pattern in refusal_patterns):
            return get_fallback_suggestion(recipient, budget_code)
        
        # Извлекаем первое предложение до точки/восклицания/вопроса
        suggestion = re.split(r'[.!?]\s', assistant_reply)[0].strip()
        suggestion = re.sub(r'\s+', ' ', suggestion)
        
        # Валидация качества ответа
        if (len(suggestion) < 10 or 
            "?" in suggestion[:30] or 
            any(x in suggestion.lower() for x in ["расскаж", "дума", "знаешь", "умеешь", "подарок", "бюджет"])):
            return get_fallback_suggestion(recipient, budget_code)
        
        if not suggestion.endswith(('.', '!', '?', '…')):
            suggestion += '.'
        
        return suggestion[:200]
    
    except Exception as e:
        return get_fallback_suggestion(recipient, budget_code) + f"\n\n⚠️ Ошибка: {str(e)[:40]}"

def get_fallback_suggestion(recipient: str, budget_code: str) -> str:
    """Хардкодные варианты на случай сбоя модели"""
    # Нормализуем ключ: "другу" → "друг"
    key = recipient.rstrip('уе').lower() if recipient not in ["программисту"] else "программист"
    
    fallbacks = {
        "друг": {
            "100": "забавная открытка с личной надписью или мини-шоколадка",
            "2500-3000": "настольная игра \"Кодenames\" или стильный чехол для телефона",
            "5000-15000": "беспроводные наушники или сертификат на квест",
            "30000-150000": "игровая приставка или билеты на концерт"
        },
        "подруг": {
            "100": "милый брелок или мини-набор конфет",
            "2500-3000": "ароматическая свеча люксового бренда или набор для скетчинга",
            "5000-15000": "стильная сумка через плечо или сертификат в спа",
            "30000-150000": "ювелирное украшение или путёвка на выходные"
        },
        "программист": {
            "100": "стикерпак с мемами про код или кружка \"Hello World\"",
            "2500-3000": "механическая клавиатура начального уровня или набор наушников",
            "5000-15000": "эргономичная мышь Logitech MX Master или подставка для монитора",
            "30000-150000": "механическая клавиатура премиум-класса или сертификат на конференцию"
        }
    }
    
    budget_text = budget_map.get(budget_code, budget_code)
    suggestion = fallbacks.get(key, fallbacks["друг"]).get(budget_code, "персонализированный подарок")
    return f"💡 Проверенный вариант для {recipient} в бюджете {budget_text}: {suggestion}"

# ===========================================
# ОБРАБОТЧИКИ (без редактирования чужих сообщений)
# ===========================================
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        "Привет! 😊 Я помогу подобрать идеальный подарок с помощью нейросети Qwen2.5.\nНажми кнопку ниже, чтобы начать:",
        reply_markup=get_main_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data == "start_flow")
def handle_start_flow(call):
    user_id = call.from_user.id
    user_state[user_id] = {"step": "choose_recipient"}
    bot.answer_callback_query(call.id)
    
    # Удаляем старое сообщение с кнопкой
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    
    bot.send_message(
        call.message.chat.id,
        "🎁 Кому будем выбирать подарок?",
        reply_markup=get_recipient_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("recipient:"))
def handle_recipient(call):
    user_id = call.from_user.id
    if user_id not in user_state or user_state[user_id].get("step") != "choose_recipient":
        bot.answer_callback_query(call.id, "Сначала нажми «Помоги выбрать подарок»", show_alert=True)
        return
    
    recipient = call.data.split(":", 1)[1]
    user_state[user_id] = {"recipient": recipient, "step": "choose_budget"}
    bot.answer_callback_query(call.id)
    
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    
    bot.send_message(
        call.message.chat.id,
        f"✅ Выбрано: подарок {recipient}\n💰 Укажи бюджет:",
        reply_markup=get_budget_keyboard()
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("budget:"))
def handle_budget(call):
    user_id = call.from_user.id
    if (user_id not in user_state or 
        user_state[user_id].get("step") != "choose_budget" or
        "recipient" not in user_state[user_id]):
        bot.answer_callback_query(call.id, "Сначала выбери получателя", show_alert=True)
        return
    
    budget_code = call.data.split(":", 1)[1]
    recipient = user_state[user_id]["recipient"]
    bot.answer_callback_query(call.id)
    
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    
    # Отправляем сообщение "думаю"
    thinking_msg = bot.send_message(
        call.message.chat.id,
        "✨ Qwen2.5 генерирует идеи подарков...\n(ожидание 8-12 секунд)"
    )
    
    # Генерация подарка
    suggestion = generate_gift_suggestion(recipient, budget_code)
    
    # Формируем ответ
    budget_text = budget_map.get(budget_code, budget_code)
    response = f"🎁 Подарок {recipient} в бюджете {budget_text}:\n\n{suggestion}"
    
    # Редактируем только своё сообщение "думаю"
    bot.edit_message_text(
        chat_id=thinking_msg.chat.id,
        message_id=thinking_msg.message_id,
        text=response
    )
    
    # Предлагаем новый подбор
    bot.send_message(
        call.message.chat.id,
        "Хочешь подобрать ещё один подарок?",
        reply_markup=get_main_keyboard()
    )
    
    # Очищаем состояние
    user_state.pop(user_id, None)

@bot.message_handler(func=lambda message: True)
def fallback_handler(message):
    bot.send_message(
        message.chat.id,
        "Нажми /start или кнопку ниже, чтобы начать подбор подарка 👇",
        reply_markup=get_main_keyboard()
    )

# ===========================================
# ЗАПУСК
# ===========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅ Бот с моделью Qwen2.5-1.5B-Instruct запущен!")
    print("💡 Первый запрос займёт 10-15 секунд (инициализация модели)")
    print("="*60 + "\n")
    bot.polling(none_stop=True)