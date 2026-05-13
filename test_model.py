import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 📌 Load trained model and tokenizer
MODEL_PATH = "./trained_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 📌 Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()
    return "sensitive" if label == 1 else "non-sensitive", confidence

# 📌 Test example
test_emails = [
    "Ваш пароль: qwerty123",
    "Доброго дня! Як справи?",
    "Документ з печаткою компанії 'Укргаз'",
    "Зустрінемось о 16:00 у кафе",
    "Реквізити компанії: IBAN UA123456789",
    "Телефон: +380971234567, Email: test@example.com",
]

# 📌 Run test e-mails through the model
for email in test_emails:
    label, confidence = predict(email)
    print(f"📩 Текст: {email}")
    print(f"   ➡ Клас: {label} (впевненність {confidence:.2f})\n")
