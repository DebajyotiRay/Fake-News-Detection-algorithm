import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_API_URL = os.getenv("HF_SPACE_API")  # e.g., https://crypticray-fake-news-detector.hf.space/run/predict

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to Fake News Detector Bot! Send me a news headline and I will check it.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text

    try:
        response = requests.post(
            HF_API_URL,
            json={"data": [user_input]},
            timeout=10
        )
        result = response.json()
        prediction = result["data"][0]
        await update.message.reply_text(prediction)

    except Exception as e:
        await update.message.reply_text("Sorry, something went wrong while checking the news.")
        print("Error:", e)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()
