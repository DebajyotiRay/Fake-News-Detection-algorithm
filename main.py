import os
import torch
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from flask import Flask
import threading

# === Load environment variables ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# === Load model and tokenizer from Hugging Face Hub ===
MODEL_REPO = "CrypticRAY/fake-news-distilbert"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Telegram Bot Logic ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a news headline and I’ll tell you if it's FAKE or REAL.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "FAKE" if pred == 1 else "REAL"
    response = f"Prediction: *{label}*\nConfidence: {confidence*100:.2f}%"
    await update.message.reply_text(response, parse_mode='Markdown')

# === Start the Telegram bot in a background thread ===
def run_telegram_bot():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), predict))
    application.run_polling()

threading.Thread(target=run_telegram_bot).start()

# === Dummy Flask server to keep Render alive ===
app = Flask(__name__)

@app.route("/")
def index():
    return "Telegram Fake News Bot is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use dynamic port from Render
    app.run(host="0.0.0.0", port=port)