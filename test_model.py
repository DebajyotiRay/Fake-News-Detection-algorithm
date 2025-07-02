from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./model"
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
        return f"Prediction: {label} (Confidence: {confidence*100:.2f}%)"

# Example prediction
example = "Aliens have landed in the White House."
print(predict(example))
