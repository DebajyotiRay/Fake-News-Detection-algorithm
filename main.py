from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import torch

# Load model and tokenizer
model_path = "model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
        label = "FAKE" if pred == 1 else "REAL"
        return f"🧠 Prediction: {label} (Confidence: {confidence*100:.2f}%)"

# Telegram start command
def start(update, context):
    update.message.reply_text("👋 Welcome! Send me a news headline and I’ll tell you if it’s FAKE or REAL.")

# Handle normal messages
def handle_message(update, context):
    user_input = update.message.text
    result = predict(user_input)
    update.message.reply_text(result)

# Main runner
def main():
    TOKEN = "8141790891:AAFEW3w_N-NdFA9lnhDS9mYHHroLfp3ct9s"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
