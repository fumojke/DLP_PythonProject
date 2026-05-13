from fastapi import FastAPI
from pymongo import MongoClient
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["email_analyzer"]
collection = db["emails"]

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# get e-mails
@app.get("/emails")
def get_emails():
    emails = collection.find().limit(50)
    return [
        {
            "id": str(email["_id"]),
            "from": email["from"],
            "to": email["to"],
            "subject": email["subject"],
            "is_sensitive": email.get("is_sensitive", False),
        }
        for email in emails
    ]

# get e-mail by id
@app.get("/emails/{email_id}")
def get_email(email_id: str):
    email = collection.find_one({"_id": ObjectId(email_id)})
    if not email:
        return {"error": "Email not found"}
    return {
        "from": email["from"],
        "to": email["to"],
        "subject": email["subject"],
        "text": email["text"],
        "is_sensitive": email.get("is_sensitive", False),
        "ai_analysis": email.get("ai_analysis", []),
    }

# Server starting
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
