from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "./trained_model"

# Загружаем модель
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()
    print(f"📨 Текст: {text}")
    print(f"📊 Предсказание: {'sensitive' if label == 1 else 'non-sensitive'} (conf={confidence:.4f})")

# Проверим 3 тестовых примера
predict_text("Логин: admin, Пароль: 1234")
predict_text("Добрый день, у нас новая встреча во вторник")
predict_text("Вот ваш секретный ключ API: XYZ123")
