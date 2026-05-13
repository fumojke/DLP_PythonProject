import asyncio
import os
import pymongo
import stanza
from aiosmtpd.controller import Controller
from email import message_from_bytes

# Подключаемся к MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["email_analyzer"]
collection = db["emails"]

# Загружаем NLP-модель Stanza для украинского языка
nlp = stanza.Pipeline(lang="uk", processors="tokenize,pos,ner")

class MailHandler:
    async def handle_DATA(self, server, session, envelope):
        message = message_from_bytes(envelope.content)

        sender = message["From"]
        recipient = message["To"]
        subject = message["Subject"]

        print(f"📩 Получено письмо от {sender}")

        # Разбираем текст и вложения
        text_content = ""
        attachments = []

        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = part.get("Content-Disposition", "")

            if content_type == "text/plain":
                text_content += part.get_payload(decode=True).decode("utf-8", errors="ignore")

            elif "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    file_data = part.get_payload(decode=True)
                    attachments.append((filename, file_data))

        # AI-анализ с помощью Stanza
        doc = nlp(text_content)

        named_entities = []
        for sentence in doc.sentences:
            for entity in sentence.ents:
                named_entities.append({"text": entity.text, "type": entity.type})

        is_sensitive = any(ent["type"] in ["ORG", "PERSON", "MISC"] for ent in named_entities)

        # Сохраняем письмо в MongoDB
        email_record = {
            "from": sender,
            "to": recipient,
            "subject": subject,
            "text": text_content,
            "ai_analysis": named_entities,
            "is_sensitive": is_sensitive,
            "attachments": [name for name, _ in attachments],
        }
        collection.insert_one(email_record)

        print(f"✅ Письмо сохранено в базе. Найдены сущности: {named_entities}")

        # Сохранение вложений
        for filename, file_data in attachments:
            os.makedirs("attachments", exist_ok=True)
            with open(os.path.join("attachments", filename), "wb") as f:
                f.write(file_data)
            print(f"📎 Вложение сохранено: {filename}")

        return "250 Message accepted"

# Start SMTP-server
controller = Controller(MailHandler(), hostname="localhost", port=1025)

async def run():
    controller.start()
    print("📬 SMTP-server started on port 1025...")
    await asyncio.sleep(3600)

asyncio.run(run())
