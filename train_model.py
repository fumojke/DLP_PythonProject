import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn
import evaluate  # Імпорт модуля для розрахунка метрік

metric = evaluate.load("accuracy")  # Завантажуємо метріку точності
# 📌 Load dataset
df = pd.read_csv("dataset.csv")
print(df["label"].value_counts(normalize=True))  # Checking class balancing

# 📌 Convert labels to numbers (sensitive = 1, non-sensitive = 0)
df["label"] = df["label"].map({"sensitive": 1, "non-sensitive": 0})

# 📌 Separate data to training and testing
test_size = max(0.2, min(10 / len(df), 0.5))  # Беремо мінімум 10 прикладів у тест
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=test_size, random_state=42, stratify=df["label"]
)

# 📌 Load tokenizer and model
MODEL_NAME = "xlm-roberta-base" # Назва моделі
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 📌 Compute class weights on train set only
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)  # Перевіримо ваги

# 📌 Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
eval_dataset = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 📌 Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 📌 Define weighted loss function
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# 📌 Train settings
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    num_train_epochs=4,  # Кількість епох для навчання
    weight_decay=0.01,
    warmup_ratio=0.1,  # Додаємо разігрів
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Берем предсказанный класс
    return metric.compute(predictions=predictions, references=labels)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

def evaluate_model(model, tokenizer, dataset):
    model.eval()  # Переводимо в режим інференса
    all_labels = []
    all_preds = []

    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()

        all_labels.append(example["label"])
        all_preds.append(label)

    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"📊 Точность на тесте: {accuracy:.4f}")

def check_confidence(model, tokenizer, dataset):
    model.eval()
    confidences = {0: [], 1: []}  # Впевненість для кожного класу

    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()

        confidences[label].append(confidence)

    print(f"📊 Середня впевненість для non-sensitive: {np.mean(confidences[0]):.4f}")
    print(f"📊 Середня впевненість для sensitive: {np.mean(confidences[1]):.4f}")

print("🔍 Кількість прикладів у тестовому наборі:")
print(eval_dataset.filter(lambda x: x["label"] == 1).num_rows, "sensitive")
print(eval_dataset.filter(lambda x: x["label"] == 0).num_rows, "non-sensitive")

# 📌 Start training
trainer.train()

evaluate_model(model, tokenizer, eval_dataset)
check_confidence(model, tokenizer, eval_dataset)
# 📌 Save model
trainer.save_model("./trained_model")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("✅ Model was trained and saved in './trained_model'")
