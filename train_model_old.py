import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.utils import class_weight
import numpy as np
import torch
import torch.nn as nn

# 📌 Load dataset
df = pd.read_csv("dataset_old.csv")
print(df["label"].value_counts(normalize=True))  # Checking class balancing

# 📌 Convert labels to numbers (sensitive = 1, non-sensitive = 0)
df["label"] = df["label"].map({"sensitive": 1, "non-sensitive": 0})

# 📌 Separate data to training and testing
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 📌 Load tokenizer and model
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Рассчитываем веса классов
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)

# Преобразуем в тензор PyTorch
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class weights:", class_weights)  # Проверим веса

# 📌 Tokenize data
def tokenize_function(texts):
    return tokenizer(texts["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
eval_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 📌 Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Определяем кастомную loss-функцию
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Убираем labels из inputs
        outputs = model(**inputs)
        logits = outputs.logits

        # Используем сбалансированные веса классов
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# 📌 Train settings
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 📌 Start training
trainer.train()

# 📌 Save model
trainer.save_model("./trained_model")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("✅ Model was trained and saved in './trained_model'")
