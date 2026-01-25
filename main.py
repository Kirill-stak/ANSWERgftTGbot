import os
import telebot
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TOKEN')

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def get_start(message):
    bot.send_message(message.chat.id, "привет")

@bot.message_handler(content_types=['text gvxe'])
def first_answer(message):
    text = message.text.lower()

    if 'програмист' in text:
        bot.send_message(message.chat.id, "ГОВНО")
    elif 'другу' in text:
        bot.send_message(message.chat.id, "ГОВНО")
    elif 'сифону' in text:
        bot.send_message(message.chat.id, "ГОВНО")
    else:
        bot.send_message(message.chat.id, "я могу покозать подарок (програмист,друг,сифонув)")





if '__main__' == __name__:
    bot.infinity_polling()