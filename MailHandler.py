import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient
from email import message_from_bytes
from aiosmtpd.controller import Controller
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from cryptography.fernet import Fernet

# Generate an encryption key (use the saved key for decryption)
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["email_analyzer"]
collection = db["emails"]

df = pd.read_csv("dataset.csv")
print(df["label"].value_counts())

# Load trained AI-model
MODEL_PATH = "./trained_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def analyze_text(text):
    if not text.strip():
        return False, 0.0  # If text is empty -> non-sensitive

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()
    return label == 1, confidence  # True for sensitive, False for non-sensitive


class MailHandler:
    async def handle_DATA(self, server, session, envelope):
        message = message_from_bytes(envelope.content)
        subject = message["Subject"]
        sender = message["From"]
        recipient = message["To"]
        body = ""

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(part.get_content_charset(), errors="ignore")
        else:
            body = message.get_payload(decode=True).decode(message.get_content_charset(), errors="ignore")

        is_sensitive, confidence = analyze_text(body)

        collection.insert_one({
            "from": sender,
            "to": recipient,
            "subject": subject,
            "text": body,
            "is_sensitive": is_sensitive,
            "confidence": confidence,
        })

        print(f"📩 E-mail from {sender} with subject '{subject}' — {'sensitive' if is_sensitive else 'non-sensitive'} (confidence {confidence:.2f})")

        # Preparing e-mail for forwarding
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject

        if is_sensitive:
            encrypted_body = cipher.encrypt(body.encode()).decode()
            msg.attach(MIMEText(f"Encrypted message: {encrypted_body}", "plain"))
        else:
            msg.attach(MIMEText(body, "plain"))

        # Sending through the local SMTP-server (localhost:1026)
        with smtplib.SMTP("localhost", 1026) as smtp:
            smtp.sendmail(sender, recipient, msg.as_string())

        print("📤 Email forwarded!")
        return "250 OK"


# Start SMTP-server
controller = Controller(MailHandler(), hostname="localhost", port=1025)
controller.start()
print("📨 SMTP-server started on localhost:1025...")

try:
    import time

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Server stopped...")
    controller.stop()
