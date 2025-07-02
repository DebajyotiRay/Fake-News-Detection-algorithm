# Fake-News-Detection-algorithm
This project is a Telegram bot that uses a fine-tuned DistilBERT model to detect whether a news headline is FAKE or REAL. It is based on natural language processing techniques using Hugging Face Transformers and PyTorch.

## Features

- Accepts text-based input from users through Telegram
- Classifies the input as FAKE or REAL
- Returns prediction along with confidence score
- Trained on merged datasets including sarcasm, fake, true, and challenging examples

## Folder Structure

```
TelegramFakeNewsBot/
├── main.py               # Bot script with Telegram integration
├── test_model.py         # Standalone test for model prediction
├── requirements.txt      # Python dependencies
├── model/                # Fine-tuned DistilBERT model and tokenizer files
```

## Setup Instructions

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Place your trained model files inside the `model/` directory.

4. Run the bot:

```
python main.py
```

The bot will begin listening on Telegram for input.

## Credits

Developed by Debajyoti Ray. Model trained using Hugging Face Transformers and deployed via python-telegram-bot.
