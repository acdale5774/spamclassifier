import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset

# Step 1: Load Dataset
dataset = load_dataset("sms_spam")  # Hugging Face SMS Spam dataset
train_data = dataset['train'].shuffle(seed=42).select(range(2000))  # Small subset for demo
val_data = dataset['test'].select(range(500))

# Step 2: Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

train_data = train_data.map(preprocess, batched=True)
val_data = val_data.map(preprocess, batched=True)

# Step 3: Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 4: DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# Step 5: Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Train for 3 epochs
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = torch.tensor(batch['label']).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Step 7: Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = torch.tensor(batch['label']).to(device)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total:.4f}")

# Step 8: Save the Model
model.save_pretrained("spam_classifier")
tokenizer.save_pretrained("spam_classifier")
