import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device):

    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, token_type_ids, item_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        item_ids = item_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids, item_ids)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Средняя потеря на эпохе: {avg_loss:.4f}")

def evaluate(model, dataloader, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids, attention_mask, token_type_ids, item_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, item_ids)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Средняя потеря на валидационном датасете: {avg_loss:.4f}")
