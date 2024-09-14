import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from recommendations.model import BERT4REC
from recomenmendations.data import KionDataset
from recommendations.utils import train_one_epoch, evaluate

def train(config_path, train_data_path, val_data_path, num_items, batch_size, epochs, lr, device):

    model = BERT4REC(config_path=config_path, num_items=num_items).to(device)


    optimizer = AdamW(model.parameters(), lr=lr)


    train_dataset = KionDataset(train_data_path)
    val_dataset = KionDataset(val_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_one_epoch(model, train_dataloader, optimizer, device)
        evaluate(model, val_dataloader, device)


    torch.save(model.state_dict(), 'bert4rec_model.pth')

if __name__ == "__main__":

    config_path = 'bert-base-uncased'
    train_data_path = 'train_data.csv'
    val_data_path = 'val_data.csv'
    num_items = 1000
    batch_size = 32
    epochs = 5
    lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train(config_path, train_data_path, val_data_path, num_items, batch_size, epochs, lr, device)
