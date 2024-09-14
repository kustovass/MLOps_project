import pandas as pd
from torch.utils.data import Dataset

class KionDataset(Dataset):

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        row = self.data.iloc[idx]
        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        token_type_ids = row['token_type_ids']
        item_ids = row['item_ids']
        labels = row['labels']

        return input_ids, attention_mask, token_type_ids, item_ids, labels
