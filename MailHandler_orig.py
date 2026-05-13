import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient
from email import message_from_bytes
from aiosmtpd.controller import Controller

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["email_analyzer"]
collection = db["emails"]
df = pd.read_csv("dataset_orig.csv")
print(df["label"].value_counts())

# Load trained AI-model
MODEL_PATH = "./trained_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
THRESHOLD = 0.53

def analyze_text(text):
    if not text.strip():
        return False, 0.0  # If text is empty -> non-sensitive

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()

        print(f"[DEBUG] Текст: {text[:100]}...")  # Покаже початок листа
        print(f"[DEBUG] Label: {label} (1=sensitive, 0=non-sensitive)")
        print(f"[DEBUG] Confidence: {confidence:.4f}")
        print(f"[DEBUG] Probs: {probs.tolist()}")  # Сирі верогідності

        print(f"📊 AI-анализ: label={label}, confidence={confidence:.2f}")  # Отладка

        # return confidence > THRESHOLD, confidence  # True for sensitive, False for non-sensitive
        return label == 1, confidence  # True for sensitive, False for non-sensitive

    except Exception as e:
        print(f"❌ Помилка під час аналізу текста: {e}")
        return False, 0.0  # In case of error e-mail will be non-sensitive

class MailHandler:
    async def handle_DATA(self, server, session, envelope):
        message = message_from_bytes(envelope.content)
        subject = message["Subject"]
        sender = message["From"]
        recipient = message["To"]
        body = ""

        # Getting text from the e-mail
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(part.get_content_charset(), errors="ignore")
        else:
            body = message.get_payload(decode=True).decode(message.get_content_charset(), errors="ignore")

        # Analyze text using AI
        is_sensitive, confidence = analyze_text(body)

        print(f"📩 E-mail text for issues analyzing: \n{body}\n")

        collection.insert_one({
            "from": sender,
            "to": recipient,
            "subject": subject,
            "text": body,
            "is_sensitive": is_sensitive,
            "confidence": confidence,
            # "attachments": [name for name, _ in attachments],
        })

        print(f"📩 E-mail received from {sender} with subject '{subject}' — {'sensitive' if is_sensitive else 'non-sensitive'}  (confidence {confidence:.2f})")
        return "250 OK"


# Start SMTP-server
controller = Controller(MailHandler(), hostname="localhost", port=1025)
controller.start()
print("📨 SMTP-server started on localhost:1025...")

try:
    import time
    while True:
        time.sleep(1)  # Support server running
except KeyboardInterrupt:
    print("🛑 Server stopped...")
    controller.stop()
